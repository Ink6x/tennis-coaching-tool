import streamlit as st
import json
import numpy as np
import faiss
from pathlib import Path
import os
from openai import OpenAI
import pickle
import hashlib
import time

# ==========================================
# 設定（Streamlit Cloud用）
# ==========================================

# OpenAI APIキーを取得（環境変数 or Streamlit Secrets）
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY') or st.secrets.get('OPENAI_API_KEY')

# APIキーが設定されているか確認
if not OPENAI_API_KEY:
    st.error("❌ OpenAI APIキーが設定されていません")
    st.warning("""
    **Streamlit Community Cloud での解決方法:**
    
    1. アプリの Settings → Secrets を開く
    2. 以下を追加:
    ```
    OPENAI_API_KEY = "sk-proj-your-api-key-here"
    ```
    3. Save をクリック
    4. アプリを Reboot
    
    **ローカル環境での解決方法:**
    
    1. `.env` ファイルを作成
    2. 以下を記入:
    ```
    OPENAI_API_KEY=sk-proj-your-api-key-here
    ```
    """)
    st.stop()

# パスワード設定（環境変数 or Streamlit Secrets or デフォルト）
APP_PASSWORD = os.getenv('APP_PASSWORD') or st.secrets.get('APP_PASSWORD', 'coaching2025')

# データファイルのパスを自動検出
def find_data_file():
    """students.json の場所を自動検出"""
    possible_paths = [
        'students.json',
        'data/students.json',
        Path(__file__).parent / 'students.json',
        Path(__file__).parent / 'data' / 'students.json'
    ]
    
    for path in possible_paths:
        p = Path(path)
        if p.exists():
            return str(p)
    
    return None

# データファイルパス
DATA_FILE = find_data_file()

if not DATA_FILE:
    st.error("❌ データファイル (students.json) が見つかりません")
    st.warning("""
    **解決方法:**
    
    1. **GitHubリポジトリに students.json を追加**:
    ```bash
    git add students.json
    git commit -m "Add students data file"
    git push
    ```
    
    2. **または data/ フォルダに配置**:
    ```
    coaching-tool/
    ├── streamlit_app.py
    └── data/
        └── students.json  ← ここに配置
    ```
    
    3. **Streamlit Cloudでアプリを Reboot**
    """)
    st.stop()

# ==========================================
# RAGシステムクラス
# ==========================================

class CoachingAssistant:
    def __init__(self, data_file=None):
        """
        コーチングアシスタントの初期化
        
        Args:
            data_file (str, optional): 生徒データのJSONファイルパス
        """
        self.data_file = data_file or DATA_FILE
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        # インデックスの保存先
        self.index_dir = Path("data")
        self.index_dir.mkdir(exist_ok=True)
        self.index_path = self.index_dir / "faiss_index.bin"
        self.chunks_path = self.index_dir / "chunks.pkl"
        
    def load_data(self):
        """JSONデータを読み込む"""
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            st.error(f"❌ データファイルが見つかりません: {self.data_file}")
            st.stop()
        except json.JSONDecodeError as e:
            st.error(f"❌ JSONファイルの形式が正しくありません: {e}")
            st.stop()
    
    def get_data_hash(self):
        """データファイルのハッシュ値を計算（変更検知用）"""
        with open(self.data_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    
    def chunk_data(self, data):
        """データをチャンクに分割（配列形式と辞書形式の両方に対応）"""
        chunks = []
        metadata = []
        
        # dataが配列の場合と辞書の場合の両方に対応
        if isinstance(data, list):
            # 配列形式の場合
            for student in data:
                student_id = student.get('id', student.get('student_id', 'unknown'))
                self._process_student(student_id, student, chunks, metadata)
        else:
            # 辞書形式の場合
            for student_id, student in data.items():
                self._process_student(student_id, student, chunks, metadata)
        
        return chunks, metadata
    
    def _process_student(self, student_id, student, chunks, metadata):
        """生徒データを処理してチャンクを作成"""
        # 基本情報チャンク
        basic_info = f"""
生徒ID: {student_id}
名前: {student.get('name', '不明')}
年齢: {student.get('age', '不明')}歳
学年: {student.get('grade', '不明')}
競技: {student.get('sport', '不明')}
"""
        chunks.append(basic_info.strip())
        metadata.append({
            'student_id': student_id,
            'type': 'basic_info',
            'name': student.get('name', '不明')
        })
        
        # 目標設定の記録
        if 'records' in student:
            for idx, record in enumerate(student['records'], 1):
                record_text = f"""
【生徒: {student.get('name', '不明')} ({student_id})】
セッション日: {record.get('date', '不明')}
現在の状況: {record.get('current_situation', '記録なし')}
目標: {record.get('goal', '記録なし')}
取り組み内容: {record.get('approach', '記録なし')}
振り返り: {record.get('reflection', '記録なし')}
コーチのメモ: {record.get('coach_notes', '記録なし')}
"""
                chunks.append(record_text.strip())
                metadata.append({
                    'student_id': student_id,
                    'type': 'record',
                    'record_index': idx,
                    'date': record.get('date', '不明'),
                    'name': student.get('name', '不明')
                })
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """テキストの埋め込みベクトルを取得"""
        try:
            text = text.replace("\n", " ")
            response = self.client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"❌ Embedding取得エラー: {e}")
            return None
    
    def get_embeddings_batch(self, texts, model="text-embedding-3-small", batch_size=1):
        """複数テキストの埋め込みを効率的に取得（超保守的レート制限対応）"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [text.replace("\n", " ") for text in batch]
            
            try:
                response = self.client.embeddings.create(input=batch, model=model)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                
                # 進捗表示
                progress = min(i + len(batch), len(texts))
                st.info(f"処理中: {progress}/{len(texts)} チャンク（約{int(progress/len(texts)*100)}%）")
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Batch embedding取得エラー: {error_msg}")
                
                # エラーがレート制限の場合、より長く待つ
                if "rate" in error_msg.lower() or "429" in error_msg:
                    st.warning("⏱️ レート制限を検出。60秒待機してから再試行します...")
                    time.sleep(60)
                    # 再試行
                    try:
                        response = self.client.embeddings.create(input=batch, model=model)
                        embeddings = [item.embedding for item in response.data]
                        all_embeddings.extend(embeddings)
                        progress = min(i + len(batch), len(texts))
                        st.success(f"✅ 再試行成功: {progress}/{len(texts)} チャンク")
                    except Exception as e2:
                        st.error(f"❌ 再試行も失敗: {e2}")
                        st.warning("""
                        **解決方法:**
                        1. 新しいAPIキーを作成
                        2. Organization/Projectの設定を確認
                        3. Tier（利用プラン）を確認
                        """)
                        return None
                else:
                    st.warning("""
                    **このエラーの原因:**
                    - APIキーの問題
                    - Project/Organizationの制限
                    
                    **解決方法:**
                    1. 新しいAPIキーを作成
                    2. Limitsページで制限を確認
                    """)
                    return None
            
            # レート制限対策（非常に保守的）
            if i + batch_size < len(texts):
                time.sleep(5.0)  # 2秒 → 5秒に変更
        
        return all_embeddings
    
    def build_index(self):
        """FAISSインデックスを構築"""
        # データ読み込み
        data = self.load_data()
        
        # チャンク化
        self.chunks, self.chunk_metadata = self.chunk_data(data)
        
        st.info(f"📊 {len(self.chunks)} 個のチャンクを処理中...")
        
        # 埋め込みベクトル取得
        embeddings = self.get_embeddings_batch(self.chunks)
        
        if not embeddings:
            st.error("❌ 埋め込みベクトルの取得に失敗しました")
            return False
        
        # NumPy配列に変換
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # FAISSインデックス構築
        dimension = embeddings_np.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(embeddings_np)
        
        # インデックスを保存
        faiss.write_index(self.index, str(self.index_path))
        
        # チャンクとメタデータを保存
        with open(self.chunks_path, 'wb') as f:
            pickle.dump({
                'chunks': self.chunks,
                'metadata': self.chunk_metadata,
                'data_hash': self.get_data_hash()
            }, f)
        
        st.success(f"✅ インデックス構築完了: {len(self.chunks)} 個のチャンク")
        return True
    
    def load_index(self):
        """保存済みのインデックスを読み込む"""
        try:
            # インデックスが存在するか確認
            if not self.index_path.exists() or not self.chunks_path.exists():
                return False
            
            # データハッシュを確認（データが更新されていないか）
            with open(self.chunks_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # saved_dataが辞書でない場合は古い形式なので削除
            if not isinstance(saved_data, dict):
                st.warning("⚠️ 古い形式のインデックスを検出。削除して再構築します。")
                self.index_path.unlink(missing_ok=True)
                self.chunks_path.unlink(missing_ok=True)
                return False
            
            current_hash = self.get_data_hash()
            if saved_data.get('data_hash') != current_hash:
                st.warning("⚠️ データファイルが更新されています。インデックスを再構築します。")
                return False
            
            # インデックスを読み込み
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = saved_data['chunks']
            self.chunk_metadata = saved_data['metadata']
            
            return True
        except Exception as e:
            st.warning(f"⚠️ インデックス読み込みエラー: {e}")
            # エラーが発生した場合は古いファイルを削除
            try:
                self.index_path.unlink(missing_ok=True)
                self.chunks_path.unlink(missing_ok=True)
            except:
                pass
            return False
    
    def search(self, query, k=5):
        """クエリに類似したチャンクを検索"""
        if self.index is None:
            st.error("❌ インデックスが構築されていません")
            return []
        
        # クエリの埋め込みベクトルを取得
        query_embedding = self.get_embedding(query)
        
        if query_embedding is None:
            return []
        
        query_vector = np.array([query_embedding], dtype=np.float32)
        
        # 検索実行
        distances, indices = self.index.search(query_vector, k)
        
        # 結果を整形
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'distance': float(distance)
                })
        
        return results
    
    def get_answer(self, query, model="gpt-4o-mini"):
        """RAGを使って質問に回答"""
        # 関連チャンクを検索
        search_results = self.search(query, k=5)
        
        if not search_results:
            return "関連する情報が見つかりませんでした。", []
        
        # コンテキストを構築
        context = "\n\n---\n\n".join([r['chunk'] for r in search_results])
        
        # プロンプト作成
        prompt = f"""あなたは子供向け(10-18歳)の1on1コーチングを行うコーチのアシスタントです。
過去の生徒データから、新しい生徒への目標設定や指導のアドバイスを提供してください。

【参考となる過去のデータ】
{context}

【質問】
{query}

【回答の指針】
- 過去の成功事例や効果的だったアプローチを参考にしてください
- 生徒の年齢や状況に応じた具体的なアドバイスを提供してください
- コーチング的な視点（傾聴、質問、目標設定）を重視してください
- 実践的で行動につながる提案を心がけてください

回答:"""
        
        try:
            # OpenAI APIで回答生成
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "あなたは経験豊富なコーチングアシスタントです。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            answer = response.choices[0].message.content
            return answer, search_results
        
        except Exception as e:
            st.error(f"❌ 回答生成エラー: {e}")
            return f"エラーが発生しました: {e}", search_results

# ==========================================
# Streamlitアプリ
# ==========================================

def main():
    st.set_page_config(
        page_title="テニスコーチング効率化ツール",
        page_icon="🎾",
        layout="wide"
    )
    
    # パスワード認証
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        st.title("🎾 テニスコーチング効率化ツール")
        st.write("過去の生徒データから、新しい目標設定の参考情報をAI検索できます")
        
        password = st.text_input("パスワードを入力してください", type="password")
        
        if st.button("ログイン"):
            if password == APP_PASSWORD:
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ パスワードが正しくありません")
        return
    
    # メインアプリ
    st.title("🎾 テニスコーチング効率化ツール")
    st.write("過去の生徒データから、新しい目標設定の参考情報をAI検索できます")
    
    # データファイル情報を表示
    st.info(f"📁 データファイル: {DATA_FILE}")
    
    # アシスタントの初期化
    if 'assistant' not in st.session_state:
        with st.spinner("📂 保存済みインデックスを読み込み中..."):
            assistant = CoachingAssistant()
            
            # インデックスを読み込む（なければ構築）
            if assistant.load_index():
                st.success(f"✅ インデックス読み込み完了: {len(assistant.chunks)} 個のチャンク")
            else:
                st.warning("⚠️ 保存済みインデックスがありません。新規構築します...")
                if assistant.build_index():
                    st.balloons()
                else:
                    st.error("❌ インデックスの構築に失敗しました")
                    return
            
            st.session_state.assistant = assistant
    
    assistant = st.session_state.assistant
    
    # サイドバー
    with st.sidebar:
        st.header("⚙️ 設定")
        
        # モデル選択
        model_name = st.selectbox(
            "使用モデル",
            ["gpt-5.1", "gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
            index=0
        )
        
        st.markdown("---")
        
        # インデックス再構築ボタン
        if st.button("🔄 インデックス再構築", type="secondary"):
            with st.spinner("インデックスを再構築中..."):
                if assistant.build_index():
                    st.success("✅ インデックスの再構築が完了しました")
                    st.rerun()
        
        # 統計情報
        st.markdown("---")
        st.markdown("### 📊 統計情報")
        st.metric("チャンク数", len(assistant.chunks))
        
        # データ情報
        data = assistant.load_data()
        if isinstance(data, list):
            st.metric("生徒数", len(data))
            total_records = sum(len(s.get('records', [])) for s in data)
        else:
            st.metric("生徒数", len(data))
            total_records = sum(len(s.get('records', [])) for s in data.values())
        st.metric("記録数", total_records)
    
    # メインコンテンツ
    st.markdown("---")
    
    # 検索入力
    st.subheader("🔍 質問を入力")
    query = st.text_area(
        "例: テニスで試合に勝てない中学生にどのような目標設定をすればいいですか？",
        height=100,
        placeholder="過去のデータから参考になる情報を検索します..."
    )
    
    # 検索実行
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 検索", type="primary")
    with col2:
        if st.button("🗑️ クリア"):
            st.rerun()
    
    if search_button and query:
        with st.spinner("🤖 AI が回答を生成中..."):
            answer, search_results = assistant.get_answer(query, model=model_name)
        
        # 回答表示
        st.markdown("---")
        st.subheader("💬 AI の回答")
        st.markdown(answer)
        
        # 参考データ表示
        st.markdown("---")
        st.subheader("📚 参考にした過去のデータ")
        
        for i, result in enumerate(search_results, 1):
            with st.expander(f"📄 参考データ {i} - {result['metadata'].get('name', '不明')} ({result['metadata'].get('type', 'unknown')})"):
                st.text(result['chunk'])
                st.caption(f"関連度スコア: {result['distance']:.4f}")

if __name__ == "__main__":
    main()

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
        """データをチャンクに分割（実際のデータ構造に合わせた実装）"""
        chunks = []
        metadata = []
        
        # データは配列形式で生徒情報を含む
        for student in data:
            student_name = student.get('name', '不明')
            
            # 基本情報チャンク
            basic_info = f"""
生徒名: {student_name}
"""
            # Visionデータをチャンク化
            if 'vision' in student:
                for vision in student['vision']:
                    if vision.get('goal'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Vision（目標設定）
目標: {vision['goal']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'vision',
                            'subtype': '目標'
                        })
                    
                    # 達成理由をチャンク化
                    if 'reasons' in vision:
                        reasons_text = []
                        reasons = vision['reasons']
                        if reasons.get('visible_self'):
                            reasons_text.append("【見える・自分】" + '; '.join(reasons['visible_self']))
                        if reasons.get('invisible_self'):
                            reasons_text.append("【見えない・自分】" + '; '.join(reasons['invisible_self']))
                        if reasons.get('visible_others'):
                            reasons_text.append("【見える・他人】" + '; '.join(reasons['visible_others']))
                        if reasons.get('invisible_others'):
                            reasons_text.append("【見えない・他人】" + '; '.join(reasons['invisible_others']))
                        
                        if reasons_text:
                            chunk_text = f"""
【生徒: {student_name}】
タイプ: Vision（達成理由）
達成したい理由: {' '.join(reasons_text)}
"""
                            chunks.append(chunk_text.strip())
                            metadata.append({
                                'student_name': student_name,
                                'type': 'vision',
                                'subtype': '達成理由'
                            })
                    
                    # ルーティンをチャンク化
                    if vision.get('routine'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Vision（ルーティン）
ルーティン: {'; '.join(vision['routine'])}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'vision',
                            'subtype': 'ルーティン'
                        })
            
            # Planデータをチャンク化
            if 'plan' in student:
                for plan in student['plan']:
                    if plan.get('goal'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Plan（計画）
計画目標: {plan['goal']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'plan',
                            'subtype': '計画目標'
                        })
                    
                    if plan.get('strengths'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Plan（武器）
武器: {plan['strengths']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'plan',
                            'subtype': '武器'
                        })
                    
                    if plan.get('challenges'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Plan（課題）
課題: {plan['challenges']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'plan',
                            'subtype': '課題'
                        })
                    
                    # ステップをチャンク化
                    if 'steps' in plan:
                        for step in plan['steps']:
                            chunk_text = f"""
【生徒: {student_name}】
タイプ: Plan（ステップ）
日付: {step.get('date', '不明')}
目標: {step.get('goal', '')}
詳細: {step.get('details', '')}
"""
                            chunks.append(chunk_text.strip())
                            metadata.append({
                                'student_name': student_name,
                                'type': 'plan',
                                'subtype': 'ステップ',
                                'date': step.get('date', '不明')
                            })
            
            # Reviewデータをチャンク化
            if 'review' in student:
                for review in student['review']:
                    if review.get('achievement_score'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Review（振り返り）
達成度評価: {review['achievement_score']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'review',
                            'subtype': '達成度'
                        })
                    
                    if review.get('quantitative'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Review（定量評価）
定量評価: {review['quantitative']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'review',
                            'subtype': '定量評価'
                        })
                    
                    if review.get('qualitative'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Review（定性評価）
定性評価: {review['qualitative']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'review',
                            'subtype': '定性評価'
                        })
                    
                    # 理由をチャンク化
                    if 'reasons' in review:
                        for reason in review['reasons']:
                            chunk_text = f"""
【生徒: {student_name}】
タイプ: Review（達成/未達成の理由）
理由: {reason}
"""
                            chunks.append(chunk_text.strip())
                            metadata.append({
                                'student_name': student_name,
                                'type': 'review',
                                'subtype': '理由'
                            })
                    
                    # 学びをチャンク化
                    if 'learnings' in review:
                        for learning in review['learnings']:
                            chunk_text = f"""
【生徒: {student_name}】
タイプ: Review（学び）
学んだこと: {learning}
"""
                            chunks.append(chunk_text.strip())
                            metadata.append({
                                'student_name': student_name,
                                'type': 'review',
                                'subtype': '学び'
                            })
                    
                    if review.get('next_goal'):
                        chunk_text = f"""
【生徒: {student_name}】
タイプ: Review（次の目標）
次の目標: {review['next_goal']}
"""
                        chunks.append(chunk_text.strip())
                        metadata.append({
                            'student_name': student_name,
                            'type': 'review',
                            'subtype': '次の目標'
                        })
            
            # Meeting Memosをチャンク化
            if 'meeting_memos' in student:
                for memo in student['meeting_memos']:
                    content = memo.get('content', '')
                    if content:
                        # 長いメモは分割
                        chunk_size = 500
                        for i in range(0, len(content), chunk_size):
                            chunk_content = content[i:i+chunk_size]
                            if len(chunk_content.strip()) > 50:  # 短すぎるチャンクは無視
                                chunk_text = f"""
【生徒: {student_name}】
タイプ: Meeting Memo
内容: {chunk_content}
"""
                                chunks.append(chunk_text.strip())
                                metadata.append({
                                    'student_name': student_name,
                                    'type': 'meeting_memo',
                                    'filename': memo.get('filename', '不明')
                                })
        
        return chunks, metadata
    
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
        progress_placeholder = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch = [text.replace("\n", " ") for text in batch]
            
            try:
                response = self.client.embeddings.create(input=batch, model=model)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                
                # 進捗表示
                progress = min(i + len(batch), len(texts))
                progress_placeholder.info(
                    f"処理中: {progress}/{len(texts)} チャンク（約{int(progress/len(texts)*100)}%）"
                )
                
                # レート制限対策のため少し待つ
                if i + batch_size < len(texts):
                    time.sleep(0.5)
                
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
                        progress_placeholder.empty()
                        return None
                else:
                    progress_placeholder.empty()
                    return None
        
        progress_placeholder.empty()
        return np.array(all_embeddings)
    
    def build_index(self):
        """FAISSインデックスを構築"""
        with st.spinner("🔨 インデックスを構築中..."):
            # データ読み込み
            data = self.load_data()
            
            # チャンク作成
            self.chunks, self.chunk_metadata = self.chunk_data(data)
            
            if not self.chunks:
                st.error("❌ チャンクが作成されませんでした")
                return False
            
            st.info(f"📄 {len(self.chunks)} 個のチャンクを作成しました")
            
            # Embeddings取得（レート制限対策：バッチサイズ1）
            embeddings = self.get_embeddings_batch(self.chunks, batch_size=1)
            
            if embeddings is None:
                st.error("❌ Embeddingの取得に失敗しました")
                return False
            
            # FAISSインデックス作成
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings)
            
            # 保存
            try:
                faiss.write_index(self.index, str(self.index_path))
                with open(self.chunks_path, 'wb') as f:
                    pickle.dump({
                        'chunks': self.chunks,
                        'metadata': self.chunk_metadata,
                        'hash': self.get_data_hash()
                    }, f)
                st.success("💾 インデックスを保存しました")
            except Exception as e:
                st.warning(f"⚠️ インデックスの保存に失敗: {e}")
            
            return True
    
    def load_index(self):
        """保存済みインデックスを読み込む"""
        try:
            if not self.index_path.exists() or not self.chunks_path.exists():
                return False
            
            # チャンクデータ読み込み
            with open(self.chunks_path, 'rb') as f:
                saved_data = pickle.load(f)
            
            # データファイルが変更されているかチェック
            current_hash = self.get_data_hash()
            if saved_data.get('hash') != current_hash:
                st.warning("⚠️ データファイルが更新されています。インデックスを再構築します...")
                return False
            
            # インデックス読み込み
            self.index = faiss.read_index(str(self.index_path))
            self.chunks = saved_data['chunks']
            self.chunk_metadata = saved_data['metadata']
            
            return True
        except Exception as e:
            st.warning(f"⚠️ インデックスの読み込みエラー: {e}")
            return False
    
    def search(self, query, k=10):
        """クエリに関連するチャンクを検索"""
        if self.index is None:
            return []
        
        # クエリのEmbedding取得
        query_embedding = self.get_embedding(query)
        if query_embedding is None:
            return []
        
        query_vec = np.array([query_embedding])
        
        # 検索実行
        distances, indices = self.index.search(query_vec, min(k, len(self.chunks)))
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx < len(self.chunks):
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'distance': float(distance),
                    'similarity': 1 / (1 + float(distance))
                })
        
        return results
    
    def get_answer(self, query, model="gpt-4o-mini"):
        """RAGを使って質問に回答"""
        # 関連チャンクを検索
        search_results = self.search(query, k=15)
        
        if not search_results:
            return "関連する情報が見つかりませんでした。", []
        
        # コンテキストを構築
        context = "【過去の生徒データから関連する情報】\n\n"
        for i, result in enumerate(search_results, 1):
            context += f"--- 関連情報 {i} (関連度: {result['similarity']:.2f}) ---\n"
            context += f"{result['chunk']}\n\n"
        
        # プロンプト作成
        system_prompt = """あなたは経験豊富なテニスコーチです。
過去の生徒の詳細なコーチング記録（目標設定、計画、振り返り、ミーティング記録）を参照できます。

【回答の原則】
1. 必ず具体的な生徒名と事例を引用すること
2. 数値データ（期間、達成度、頻度など）を明示すること
3. 成功例だけでなく、失敗例や困難だった点も含めること
4. 複数の生徒の事例を比較・統合して回答すること
5. 推測ではなく、データに基づいた事実のみを述べること

【回答フォーマット】
## 結論（端的に）
[質問への直接的な回答を1-2行で]

## 具体的事例
**[生徒名]の事例:**
- 目標: [具体的な目標]
- 期間: [X週間/Xヶ月]
- アプローチ: [具体的な方法]
- 結果: [達成度・学び]
- 重要ポイント: [成功/失敗の要因]

（2-3名の事例を記載）

## データから見える傾向
- [複数事例から見える共通点]
- [効果的だったアプローチ]
- [避けるべき落とし穴]

## 推奨事項
1. [具体的なアクション1]（根拠: [生徒名]の事例）
2. [具体的なアクション2]（根拠: [生徒名]の事例）
3. [具体的なアクション3]（根拠: [生徒名]の事例）"""
        
        prompt = f"{context}\n\n質問: {query}"
        
        try:
            # OpenAI APIで回答生成
            if model.lower().startswith("gpt-5") or model.lower().startswith("o1"):
                # o1/gpt-5シリーズ用の設定
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "user", "content": f"{system_prompt}\n\n{prompt}"}
                    ],
                    max_completion_tokens=8000
                )
            else:
                # 通常のGPTモデル用
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=2000,
                    temperature=0.7
                )
            
            answer = response.choices[0].message.content
            
            if not answer or answer.strip() == "":
                answer = "回答が生成されませんでした。"
            
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
        
        st.info("パスワード: coaching2025")
        return
    
    # メインアプリ
    st.title("🎾 テニスコーチング効率化ツール")
    st.write("過去の生徒データから、新しい目標設定の参考情報をAI検索できます")
    
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
        st.metric("生徒数", len(data))
        
        # 生徒一覧（上位10名）
        st.subheader("生徒一覧")
        for student in data[:10]:
            with st.expander(student.get('name', '不明')):
                st.write(f"Vision: {len(student.get('vision', []))}件")
                st.write(f"Plan: {len(student.get('plan', []))}件")
                st.write(f"Review: {len(student.get('review', []))}件")
                st.write(f"MTGメモ: {len(student.get('meeting_memos', []))}件")
    
    # サンプル質問ボタン
    st.header("🔍 質問を入力")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 3ヶ月で関東大会に出場するには?"):
            st.session_state.query = "3ヶ月で関東大会に出場するためには？"
    with col2:
        if st.button("📌 バックハンド強化の成功例は?"):
            st.session_state.query = "12歳でバックハンドを強化したい生徒の成功例を教えて"
    with col3:
        if st.button("📌 自信をつける方法は?"):
            st.session_state.query = "自信をつけるための効果的なアプローチを教えて"
    
    # 検索入力
    query = st.text_area(
        "質問を入力してください",
        value=st.session_state.get('query', ''),
        height=120,
        placeholder="例: テニスで試合に勝てない中学生にどのような目標設定をすればいいですか？"
    )
    
    # 検索実行
    col1, col2 = st.columns([1, 5])
    with col1:
        search_button = st.button("🔍 検索", type="primary")
    with col2:
        if st.button("🗑️ クリア"):
            if 'query' in st.session_state:
                del st.session_state.query
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
        st.subheader("📚 参考にした過去のデータ（上位10件）")
        
        for i, result in enumerate(search_results[:10], 1):
            student_name = result['metadata'].get('student_name', '不明')
            data_type = result['metadata'].get('type', 'unknown')
            subtype = result['metadata'].get('subtype', '')
            similarity = result['similarity']
            
            with st.expander(
                f"{i}. {student_name} - {data_type}: {subtype} (関連度: {similarity:.2%})"
            ):
                st.text(result['chunk'])
                st.caption(f"関連度スコア: {result['distance']:.4f}")
    
    # 付帯情報
    st.markdown("---")
    st.info(f"📁 データファイル: {DATA_FILE}")

if __name__ == "__main__":
    main()

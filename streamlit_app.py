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
        """データを生徒ごとにまとめた大きなチャンクに分割（最適化版）"""
        chunks = []
        metadata = []
        
        # 各生徒のデータを1-2個の大きなチャンクにまとめる
        for student in data:
            student_name = student.get('name', '不明')
            
            # === チャンク1: Vision + Plan の統合チャンク ===
            vision_plan_content = f"【生徒名: {student_name}】\n\n"
            
            # Vision情報をまとめて追加
            if 'vision' in student and student['vision']:
                vision_plan_content += "■ Vision（目標設定）\n"
                for idx, vision in enumerate(student['vision'], 1):
                    if vision.get('goal'):
                        vision_plan_content += f"  目標{idx}: {vision['goal']}\n"
                    
                    if 'reasons' in vision:
                        reasons = vision['reasons']
                        if any([reasons.get('visible_self'), reasons.get('invisible_self'), 
                               reasons.get('visible_others'), reasons.get('invisible_others')]):
                            vision_plan_content += "  達成したい理由:\n"
                            if reasons.get('visible_self'):
                                vision_plan_content += f"    見える・自分: {'; '.join(reasons['visible_self'])}\n"
                            if reasons.get('invisible_self'):
                                vision_plan_content += f"    見えない・自分: {'; '.join(reasons['invisible_self'])}\n"
                            if reasons.get('visible_others'):
                                vision_plan_content += f"    見える・他人: {'; '.join(reasons['visible_others'])}\n"
                            if reasons.get('invisible_others'):
                                vision_plan_content += f"    見えない・他人: {'; '.join(reasons['invisible_others'])}\n"
                    
                    if vision.get('routine'):
                        vision_plan_content += f"  ルーティン: {'; '.join(vision['routine'])}\n"
                    vision_plan_content += "\n"
            
            # Plan情報をまとめて追加
            if 'plan' in student and student['plan']:
                vision_plan_content += "■ Plan（計画）\n"
                for idx, plan in enumerate(student['plan'], 1):
                    if plan.get('goal'):
                        vision_plan_content += f"  計画目標{idx}: {plan['goal']}\n"
                    if plan.get('strengths'):
                        vision_plan_content += f"  武器: {plan['strengths']}\n"
                    if plan.get('challenges'):
                        vision_plan_content += f"  課題: {plan['challenges']}\n"
                    
                    if 'steps' in plan and plan['steps']:
                        vision_plan_content += "  ステップ:\n"
                        for step in plan['steps']:
                            vision_plan_content += f"    - {step.get('date', '不明')}: {step.get('goal', '')} "
                            if step.get('details'):
                                vision_plan_content += f"({step['details']})"
                            vision_plan_content += "\n"
                    vision_plan_content += "\n"
            
            # Vision + Planのチャンクを追加（内容がある場合のみ）
            if len(vision_plan_content.strip()) > 50:
                chunks.append(vision_plan_content.strip())
                metadata.append({
                    'student_name': student_name,
                    'type': 'vision_plan',
                    'content_type': 'Vision+Plan統合'
                })
            
            # === チャンク2: Review + Meeting Memos の統合チャンク ===
            review_memo_content = f"【生徒名: {student_name}】\n\n"
            
            # Review情報をまとめて追加
            if 'review' in student and student['review']:
                review_memo_content += "■ Review（振り返り）\n"
                for idx, review in enumerate(student['review'], 1):
                    review_memo_content += f"  振り返り{idx}:\n"
                    
                    if review.get('achievement_score'):
                        review_memo_content += f"    達成度評価: {review['achievement_score']}\n"
                    if review.get('quantitative'):
                        review_memo_content += f"    定量評価: {review['quantitative']}\n"
                    if review.get('qualitative'):
                        review_memo_content += f"    定性評価: {review['qualitative']}\n"
                    
                    if 'reasons' in review and review['reasons']:
                        review_memo_content += "    理由:\n"
                        for reason in review['reasons'][:3]:  # 最初の3つまで
                            review_memo_content += f"      - {reason}\n"
                        if len(review['reasons']) > 3:
                            review_memo_content += f"      （他{len(review['reasons'])-3}件）\n"
                    
                    if 'learnings' in review and review['learnings']:
                        review_memo_content += "    学び:\n"
                        for learning in review['learnings'][:3]:  # 最初の3つまで
                            review_memo_content += f"      - {learning}\n"
                        if len(review['learnings']) > 3:
                            review_memo_content += f"      （他{len(review['learnings'])-3}件）\n"
                    
                    if review.get('next_goal'):
                        review_memo_content += f"    次の目標: {review['next_goal']}\n"
                    review_memo_content += "\n"
            
            # Meeting Memos情報をまとめて追加（要約版）
            if 'meeting_memos' in student and student['meeting_memos']:
                review_memo_content += "■ Meeting Memos（ミーティング記録）\n"
                for idx, memo in enumerate(student['meeting_memos'][:5], 1):  # 最新5件まで
                    content = memo.get('content', '')
                    if content:
                        # 内容を要約（最初の500文字まで）
                        summary = content[:500]
                        if len(content) > 500:
                            summary += "..."
                        review_memo_content += f"  MTG{idx} ({memo.get('filename', '不明')}):\n"
                        review_memo_content += f"    {summary}\n\n"
                
                if len(student['meeting_memos']) > 5:
                    review_memo_content += f"  （他{len(student['meeting_memos'])-5}件のミーティング記録）\n"
            
            # Review + Meeting Memosのチャンクを追加（内容がある場合のみ）
            if len(review_memo_content.strip()) > 50:
                chunks.append(review_memo_content.strip())
                metadata.append({
                    'student_name': student_name,
                    'type': 'review_memo',
                    'content_type': 'Review+MeetingMemo統合'
                })
        
        return chunks, metadata
    
    def get_embedding(self, text, model="text-embedding-3-small"):
        """テキストの埋め込みベクトルを取得"""
        try:
            text = text.replace("\n", " ")
            # 長すぎるテキストは切り詰め（8191トークン制限対策）
            if len(text) > 8000:
                text = text[:8000]
            response = self.client.embeddings.create(input=[text], model=model)
            return response.data[0].embedding
        except Exception as e:
            st.error(f"❌ Embedding取得エラー: {e}")
            return None
    
    def get_embeddings_batch(self, texts, model="text-embedding-3-small", batch_size=5):
        """複数テキストの埋め込みを効率的に取得"""
        all_embeddings = []
        progress_placeholder = st.empty()
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            # 各テキストの長さを制限
            batch = [text.replace("\n", " ")[:8000] for text in batch]
            
            try:
                response = self.client.embeddings.create(input=batch, model=model)
                embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(embeddings)
                
                # 進捗表示
                progress = min(i + len(batch), len(texts))
                progress_placeholder.info(
                    f"処理中: {progress}/{len(texts)} チャンク"
                )
                
                # レート制限対策のため少し待つ（チャンク数が少ないので短めで良い）
                if i + batch_size < len(texts):
                    time.sleep(0.2)
                
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Batch embedding取得エラー: {error_msg}")
                
                # エラーがレート制限の場合、待機して再試行
                if "rate" in error_msg.lower() or "429" in error_msg:
                    st.warning("⏱️ レート制限を検出。30秒待機してから再試行します...")
                    time.sleep(30)
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
            
            # Embeddings取得（バッチサイズを大きく）
            embeddings = self.get_embeddings_batch(self.chunks, batch_size=5)
            
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
    
    def search(self, query, k=5):
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
        # 関連チャンクを検索（チャンク数が少ないので上位5個で十分）
        search_results = self.search(query, k=5)
        
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
1. 具体的な生徒名と事例を引用すること
2. 特定の競技や学年、年齢、成績など）が一致する場合は、それらの情報を明示すること
3. 成功例だけでなく、課題や改善点も含めること
4. 実在の過去データに基づいた具体的なアドバイスをすること
5. 推測ではなく、データに基づいた事実を述べること

【回答フォーマット】
## 結論
[質問への直接的な回答を2-3行で]

## 具体的な参考事例
[過去の生徒データから関連する事例を2-3つ紹介]

## 推奨するアプローチ
[データから見える効果的な方法を箇条書きで3-5個]

## 注意点
[避けるべきことや気をつける点を2-3個]"""
        
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
        with st.spinner("📂 初期化中..."):
            assistant = CoachingAssistant()
            
            # インデックスを読み込む（なければ構築）
            if assistant.load_index():
                st.success(f"✅ インデックス読み込み完了: {len(assistant.chunks)} 個のチャンク")
            else:
                st.info("⚠️ 初回起動のため、インデックスを構築します...")
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
        st.metric("チャンクあたりの平均文字数", 
                  int(sum(len(c) for c in assistant.chunks) / len(assistant.chunks)) if assistant.chunks else 0)
        
        # データ情報
        data = assistant.load_data()
        st.metric("生徒数", len(data))
        
        # 生徒一覧（上位5名のみ表示）
        st.subheader("生徒一覧（上位5名）")
        for student in data[:5]:
            st.write(f"• {student.get('name', '不明')}")
        if len(data) > 5:
            st.write(f"  他{len(data)-5}名")
    
    # サンプル質問ボタン
    st.header("🔍 質問を入力")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📌 16歳以下で4C大会ベスト4を目指すには?"):
            st.session_state.query = "16歳以下の4C大会でベスト4に入るための効果的な練習方法と目標設定を教えてください"
    with col2:
        if st.button("📌 試合の入りを改善する方法は?"):
            st.session_state.query = "試合の序盤でミスが多い生徒への指導方法を教えてください"
    with col3:
        if st.button("📌 メンタル強化のアプローチは?"):
            st.session_state.query = "プレッシャーに弱い生徒のメンタル強化方法を教えてください"
    
    # 検索入力
    query = st.text_area(
        "質問を入力してください",
        value=st.session_state.get('query', ''),
        height=100,
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
        if search_results:
            st.markdown("---")
            st.subheader("📚 参考にした過去のデータ")
            
            for i, result in enumerate(search_results, 1):
                student_name = result['metadata'].get('student_name', '不明')
                content_type = result['metadata'].get('content_type', '')
                similarity = result['similarity']
                
                with st.expander(
                    f"{i}. {student_name} - {content_type} (関連度: {similarity:.1%})"
                ):
                    # チャンク内容を整形して表示
                    content_lines = result['chunk'].split('\n')
                    for line in content_lines[:30]:  # 最初の30行まで表示
                        if line.strip():
                            st.text(line)
                    if len(content_lines) > 30:
                        st.text("...")
                        st.caption(f"（全{len(content_lines)}行）")
    
    # 付帯情報
    st.markdown("---")
    st.caption(f"📁 データファイル: {DATA_FILE} | 💾 チャンク数: {len(assistant.chunks)}")

if __name__ == "__main__":
    main()

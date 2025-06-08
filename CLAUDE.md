# CLAUDE.md

このファイルは、このリポジトリでコードを扱う際のClaude Code（claude.ai/code）へのガイダンスを提供します。

## プロジェクト概要

マルチプロバイダー対応のエンタープライズグレードチャットアプリケーション。OpenAI APIとAWS Bedrockの両方をサポートし、Strands agentフレームワークとMCP（Model Context Protocol）ツールシステムを統合した高度な対話型AIインターフェースを提供。

## 開発環境のセットアップ

### 前提条件
- Python 3.12以上
- OpenAI APIキー（OpenAIモデル使用時）
- AWS アカウントとBedrock アクセス権限（Bedrockモデル使用時）
- Docker（一部MCPツール用）

### インストールと実行

```bash
# 依存関係のインストール
pip install -e .
# または uv を使用
uv pip install -e .

# 環境設定ファイルの作成
cp .env.example .env
# .envファイルを編集してAPIキーを設定

# Streamlitアプリケーションの起動
streamlit run app.py
```

### 環境変数の設定

`.env`ファイルで以下を設定：
```bash
# OpenAI API設定
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# プロバイダー選択（openai または bedrock）
DEFAULT_PROVIDER=openai

# AWS設定（Bedrock使用時）
# AWS_PROFILE=default
# AWS_REGION=ap-northeast-1

# 開発モード
DEV=false
```

## アーキテクチャ詳細

### コア技術スタック

- **Streamlit**: WebUIフレームワーク
- **Strands Agent Framework v0.1.6+**: LLMインタラクションとツール管理
- **マルチプロバイダー対応**:
  - **OpenAI API**: GPT-4o、GPT-4o Mini、GPT-4.1
  - **AWS Bedrock**: Claude、Amazon Nova
- **MCP（Model Context Protocol）**: 外部ツール統合プロトコル
- **nest_asyncio**: 同期環境での非同期処理実現

### アプリケーションフロー

1. **初期化フェーズ**
   - 設定ファイル（config/*.json）の読み込み
   - 環境変数の検証（OpenAI APIキーまたはAWS認証）
   - Streamlit UIの構築

2. **ユーザーインタラクションフェーズ**
   - プロバイダー選択（OpenAI/Amazon Bedrock）
   - モデル選択（プロバイダー別）
   - MCPツール選択
   - チャット履歴の管理

3. **処理フェーズ**
   - プロバイダー別モデル初期化
   - MCPツールの動的ロード
   - エージェントの実行
   - ストリーミングレスポンス処理

4. **結果表示フェーズ**
   - リアルタイムレスポンス表示
   - ツール使用の可視化
   - チャット履歴の更新と保存

### 重要な実装パターン

#### モデル初期化の抽象化
```python
def initialize_model(provider: str, model_id: str, config: dict, enable_cache: dict):
    if provider == "openai":
        return OpenAIModel(
            client_args={"api_key": api_key},
            model_id=model_id,
            params={"max_tokens": ..., "temperature": ...}
        )
    elif provider == "bedrock":
        return BedrockModel(
            model_id=model_id,
            boto_session=boto3.Session(region_name=...),
            cache_prompt=...,
            cache_tools=...
        )
```

#### プロバイダー別機能管理
```python
def get_model_capabilities(provider: str, model_id: str, config: dict) -> dict:
    # プロバイダーとモデルに応じた機能を返す
    # - image_support: 画像入力対応
    # - cache_support: キャッシュ機能（Bedrockのみ）
    # - streaming: ストリーミング対応
    # - tools: ツール使用対応
```

## 設定ファイル詳細

### config/config.json
```json
{
    "chat_history_dir": "chat_history",
    "mcp_config_file": "config/mcp.json",
    "providers": {
        "openai": {
            "models": {
                "gpt-4o": {
                    "display_name": "GPT-4o (最新・画像対応)",
                    "image_support": true,
                    "max_tokens": 16384,
                    "cost_per_1k_tokens": {...}
                },
                "gpt-4.1": {
                    "display_name": "GPT-4.1 (2025年4月リリース)",
                    "image_support": true,
                    "max_tokens": 16384,
                    "cost_per_1k_tokens": {...}
                }
            }
        },
        "bedrock": {
            "region": "ap-northeast-1",
            "models": {
                "us.anthropic.claude-3-5-sonnet-20241022-v2:0": {
                    "display_name": "Claude 3.5 Sonnet",
                    "cache_support": [],
                    "image_support": true
                }
            }
        }
    }
}
```

## サポートモデル一覧

### OpenAI モデル
- **GPT-4o**: 最新の統合モデル、画像入力対応
- **GPT-4o Mini**: 高速・低コスト版、画像入力対応  
- **GPT-4.1**: 2025年4月リリースの新モデル、画像入力対応

### AWS Bedrock モデル
- **Claude Sonnet 4**: 最新版、プロンプトキャッシング対応
- **Claude 3.7 Sonnet**: プロンプトキャッシング対応
- **Claude 3.5 Sonnet**: スタンダード版
- **Claude 3.5 Haiku**: 高速版
- **Amazon Nova Premier/Pro/Lite/Micro**: Amazon独自モデル

## UI/UX仕様

### サイドバー機能
- **プロバイダー選択**: OpenAI/Amazon Bedrockの切り替え
- **モデル選択**: プロバイダー別のモデルリスト
- **プロバイダー別設定**:
  - OpenAI: Temperature調整スライダー
  - Bedrock: プロンプトキャッシュトグル
- **MCPツール選択**: 複数選択可能
- **チャット履歴**: 最新20件の表示と選択
- **新規チャット**: 新しい会話の開始

### メインチャット領域
- **入力エリア**: テキスト入力とファイルアップロード
- **画像プレビュー**: 対応モデルでの画像表示
- **レスポンス表示**: ストリーミング対応のマークダウン表示
- **ツール使用表示**: エキスパンダーでJSON形式表示

## エラーハンドリング

### API認証エラー
```python
# OpenAI
if not api_key:
    st.error("🔑 OPENAI_API_KEY が設定されていません。.env ファイルを確認してください。")
    st.stop()

# Bedrock
# boto3が自動的にAWS認証を処理
```

### モデル制限の処理
- OpenAIモデルの`max_tokens`制限を適切に設定
- 画像非対応モデルでの警告表示
- プロバイダー別の機能差異を明確化

## 開発時の注意事項

### マルチプロバイダー対応
- プロバイダー切り替え時の状態管理に注意
- 各プロバイダーの制限事項を考慮
- 機能の有無を適切にUIに反映

### APIキー管理
- OpenAI APIキーは環境変数で管理
- AWS認証は標準的な方法（~/.aws/credentials、IAMロール等）で対応
- .envファイルは絶対にコミットしない

### パフォーマンス最適化
- Bedrockのプロンプトキャッシングを活用
- OpenAIのストリーミングレスポンスを効率的に処理
- 大きな画像ファイルは自動的にbase64エンコード

### デバッグとトラブルシューティング
- `DEV`環境変数でデバッグモード有効化
- プロバイダー別のエラーメッセージを適切に表示
- Streamlitのセッション状態を活用した状態管理
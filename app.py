from flask import Flask, request, jsonify
from langchain_core.documents import Document

from spark_api import SparkAPI
from vector_store import VectorStore, process_text, delete_text_by_metadata
from config import Config

app = Flask(__name__)
spark = SparkAPI()
vector_store = VectorStore()


@app.route('/ask', methods=['POST'])
def ask_question():
    """RAG问答接口"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        query = data['query']
        top_k = int(data.get('top_k', 3))  # 默认获取3条相关结果

        # 1. 向量检索
        results = vector_store.similarity_search_with_score(
            query=query,
            k=top_k
        )

        # 如果没有检索到结果，返回提示
        if not results:
            return jsonify({"answer": "暂无相关数据，无法回答问题。"})

        # 2. 拼接上下文
        context = "\n".join([doc.page_content for doc, _ in results])

        # 3. 构造提示词
        prompt = f"基于以下上下文回答问题：\n{context}\n\n问题：{query}\n答案："

        # 4. 调用大模型
        response = spark.get_response(prompt)
        return jsonify({"answer": response})

    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"问答失败: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/add', methods=['POST'])
def add_text():
    """添加文本到向量库（支持单条）"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    try:
        # 处理元数据
        metadata = data.get('metadata', {})
        # 添加必要验证（如metadata必须是字典）
        if not isinstance(metadata, dict):
            return jsonify({"error": "metadata must be a dictionary"}), 400

        # 处理文本
        process_text(text=data['text'], metadata=metadata)
        return jsonify({"status": "success", "count": 1}), 201

    except Exception as e:
        app.logger.error(f"添加文本失败: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500



# 如果要支持批量添加，可以修改/add接口：
@app.route('/add_batch', methods=['POST'])
def add_batch():
    data = request.get_json()
    if not data or 'texts' not in data:
        return jsonify({"error": "Missing 'texts' array"}), 400

    try:
        texts = data['texts']
        metadatas = data.get('metadatas', [{}] * len(texts))

        # 转换为Document列表
        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]

        vector_store.add_documents(docs)
        vector_store.persist()
        return jsonify({"status": "success", "count": len(docs)})

    except Exception as e:
        app.logger.error(f"批量添加失败: {str(e)}")
        return jsonify({"error": "Batch add failed"}), 500

@app.route('/search', methods=['POST'])
def search_text():
    """相似文本搜索"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        top_k = int(data.get('top_k', 5))
        # 执行搜索
        results = vector_store.similarity_search(
            query=data['query'],
            k=top_k,
            filter=data.get('filter')  # 支持元数据过滤
        )

        # 格式化结果
        formatted = [{
            "text": doc.page_content,
            "metadata": doc.metadata,
            # "score": doc.score  # 不支持得分
        } for doc in results]

        return jsonify({"results": formatted})

    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"搜索失败: {str(e)}")
        return jsonify({"error": "Search failed"}), 500


@app.route('/delete', methods=['POST'])
def delete_text():
    """根据元数据删除文本"""
    data = request.get_json()
    if not data or 'filter' not in data:
        return jsonify({"error": "Missing 'filter' field"}), 400

    try:
        # 执行删除
        delete_text_by_metadata(filter=data['filter'])
        return jsonify({"status": "success"})

    except Exception as e:
        app.logger.error(f"删除失败: {str(e)}")
        return jsonify({"error": "Delete failed"}), 500


if __name__ == '__main__':
    app.run(port=5000, debug=True)

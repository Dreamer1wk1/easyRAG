import atexit
import json

from flask import Flask, request, jsonify
from langchain_core.documents import Document

from nacos_service import NacosService
from spark_api import SparkAPI
from vector_store import VectorStore, process_text, delete_text_by_metadata
from reranker_service import get_reranker_service

app = Flask(__name__)
spark = SparkAPI()
vector_store = VectorStore()

# 初始化并注册Nacos服务
nacos_service = NacosService()

from flask import Response, stream_with_context, request


@app.route('/ask-stream', methods=['POST'])  # 新建流式接口
def stream_qa():
    """流式问答接口"""
    data = request.get_json()

    # 参数校验（与原有逻辑一致）
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        query = data['query']
        top_k = int(data.get('top_k', 3))
        function = data.get('function', 'qa')

        # 流式生成器核心逻辑
        def generate():
            # 向量检索部分保持同步
            if function == 'qa':
                results = vector_store.similarity_search_with_score(query=query, k=top_k)
                if not results:
                    yield "data: 暂无相关数据，无法回答问题。\n\n"
                    return
                context = "\n".join([doc.page_content for doc, _ in results])
                prompt = f"基于以下上下文回答问题：\n{context}\n\n问题：{query}\n答案："
            elif function == 'translate':
                prompt = f"请将以下内容翻译成中文：\n{query}\n翻译："
            else:
                yield "data: {\"error\": \"Invalid function\"}\n\n"
                return

            # 调用支持流式输出的 LLM 接口（假设 spark.stream_response 返回生成器）
            for chunk in spark.stream_response(prompt):  # 需确保该方法是流式API
                yield f"data: {json.dumps({'answer': chunk})}\n\n"  # SSE 格式

        return Response(stream_with_context(generate()), mimetype="text/event-stream")

    except Exception as e:
        app.logger.error(f"流式请求失败: {str(e)}")
        return jsonify({"error": "Internal server error"}), 500


@app.route('/ask', methods=['POST'])
def ask_question():
    """支持问答和翻译的接口"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        query = data['query']
        top_k = int(data.get('top_k', 3))  # 默认获取3条相关结果
        function = data.get('function', 'qa')  # 默认功能是问答

        if function == 'qa':
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

        elif function == 'translate':
            # 翻译功能
            prompt = f"请将以下内容翻译成中文：\n{query}\n翻译："

        else:
            return jsonify({"error": "Invalid function. Supported functions are 'qa' and 'translate'."}), 400

        # 4. 调用大模型
        response = spark.get_response(prompt)
        return jsonify({"answer": response})

    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"请求失败: {str(e)}")
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
    """相似文本搜索（返回相关性分数）"""
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    try:
        top_k = int(data.get('top_k', 5))
        
        # 使用 similarity_search_with_score，返回 (doc, distance) 对
        # Chroma 默认使用 L2 距离，距离越小越相似
        results_with_scores = vector_store.similarity_search_with_score(
            query=data['query'],
            k=top_k,
            filter=data.get('filter')  # 支持元数据过滤
        )

        # 将距离转换为相关性分数 (0-1，越大越相关)
        # 使用 max(0, 1 - distance/2) 确保分数在 0-1 范围
        # L2 距离通常在 0-2 之间（归一化向量的情况下）
        formatted = []
        for doc, distance in results_with_scores:
            # 距离转相关性：distance=0 -> score=1, distance=2 -> score=0
            score = max(0.0, min(1.0, 1.0 - distance / 2.0))
            formatted.append({
                "text": doc.page_content,
                "metadata": {**doc.metadata, "score": round(score, 4)},
                "score": round(score, 4)
            })

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


@app.route('/rerank', methods=['POST'])
def rerank():
    """
    重排序接口
    使用 BGE-Reranker 模型对文档进行精排
    
    Request Body:
    {
        "query": "查询文本",
        "documents": ["文档1", "文档2", ...],
        "topK": 5  // 可选，默认5
    }
    
    Response:
    {
        "results": [
            {"index": 0, "score": 0.95, "text": "文档1"},
            {"index": 2, "score": 0.87, "text": "文档3"},
            ...
        ]
    }
    """
    data = request.get_json()
    
    # 参数校验
    if not data:
        return jsonify({"error": "Missing request body"}), 400
    if 'query' not in data:
        return jsonify({"error": "Missing 'query' field"}), 400
    if 'documents' not in data:
        return jsonify({"error": "Missing 'documents' field"}), 400
    
    try:
        query = data['query']
        documents = data['documents']
        top_k = int(data.get('topK', 5))
        
        # 参数验证
        if not isinstance(documents, list):
            return jsonify({"error": "'documents' must be a list"}), 400
        if not documents:
            return jsonify({"results": []})
        
        # 调用 Reranker 服务
        reranker_service = get_reranker_service()
        results = reranker_service.rerank(
            query=query,
            documents=documents,
            top_k=top_k
        )
        
        return jsonify({"results": results})
        
    except ValueError as e:
        return jsonify({"error": f"Invalid parameter: {str(e)}"}), 400
    except Exception as e:
        app.logger.error(f"重排序失败: {str(e)}")
        return jsonify({"error": "Rerank failed", "detail": str(e)}), 500

# 用.\.venv\Scripts\python.exe app.py启动
if __name__ == '__main__':
    nacos_service = NacosService()
    if nacos_service.register():
        app.logger.info("Nacos 注册成功，启动 Flask 服务...")
    else:
        app.logger.info("⚠️ Nacos 注册失败，仍然启动 Flask 服务...")
    atexit.register(nacos_service.deregister)
    app.run(host='0.0.0.0', port=5000)

import streamlit as st
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import json

# ======================
# 资源加载（只执行一次）
# ======================

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

model = load_model()

@st.cache_data
def load_metadata():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    metadata_path = os.path.join(data_dir, "metadata_with_embeddings.json")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f)

metadata = load_metadata()

# 提取文件名和嵌入向量
filenames = []
embeddings = []
for filename, data in metadata.items():
    embeddings.append(data["embedding"])
    filenames.append(filename)

embedding_matrix = np.array(embeddings, dtype="float32")

@st.cache_resource
def load_index(_embedding_matrix):
    dimension = _embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(_embedding_matrix)
    return index

index = load_index(embedding_matrix)

# ======================
# 页面初始化
# ======================

st.title("微信表情包语义搜索")
st.write("输入描述，找到最匹配的表情包！")

# 初始化 session_state
if "result_images" not in st.session_state:
    st.session_state.result_images = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "query_embedding" not in st.session_state:
    st.session_state.query_embedding = None

# 按钮回调函数
def on_prev_click():
    st.session_state.current_index = max(0, st.session_state.current_index - 1)

def on_next_click():
    st.session_state.current_index = min(len(st.session_state.result_images) - 1, st.session_state.current_index + 1)

# 输入框
query = st.text_input("输入描述", placeholder="例如：我装作没看到老板@我的消息")

# 缓存查询 embedding
if query and (query != st.session_state.last_query or st.session_state.query_embedding is None):
    with st.spinner("正在编码查询内容..."):
        new_embedding = model.encode(query).astype("float32")
        st.session_state.query_embedding = new_embedding
        st.session_state.last_query = query

# 执行搜索逻辑（仅当有新查询时）
if st.session_state.query_embedding is not None:
    distances, indices = index.search(np.array([st.session_state.query_embedding]), k=10)
    st.session_state.result_images = [filenames[i] for i in indices[0]]
    if query != st.session_state.get("prev_query_for_result", ""):
        st.session_state.current_index = 0
        st.session_state.prev_query_for_result = query

# 显示结果图片
script_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(script_dir, "static", "images")

if st.session_state.result_images:
    result_image = os.path.join(image_dir, st.session_state.result_images[st.session_state.current_index])
    st.image(result_image,
             caption=f"结果 {st.session_state.current_index + 1}/{len(st.session_state.result_images)}",
             use_container_width=False,
             width=300)

    # 上一张和下一张按钮
    col1, col2 = st.columns(2)
    with col1:
        st.button("上一张", on_click=on_prev_click)
    with col2:
        st.button("下一张", on_click=on_next_click)
else:
    if query:  # 只有在用户输入后才显示提示信息
        st.write("没有找到匹配的表情包，请尝试其他描述！")
r
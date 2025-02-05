import streamlit as st
import cv2
import numpy as np
import asyncio
import threading
from io import BytesIO
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.user import User
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from clarifai.modules.css import ClarifaiStreamlitCSS
from dotenv import load_dotenv
from typing import List, Iterator, Tuple

load_dotenv()

st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

if "selected_video" not in st.session_state:
    st.session_state.selected_video = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "video_loaded" not in st.session_state:
    st.session_state.video_loaded = False

auth = ClarifaiAuthHelper.from_streamlit(st)
pat = auth._pat
userDataObject = auth.get_user_app_id_proto()
user_id = userDataObject.user_id
app_id = userDataObject.app_id

input_obj = User(user_id=user_id).app(app_id=app_id, pat=pat).inputs()
page_no = 1
per_page = 25
page_of_inputs = list(input_obj.list_inputs(page_no=page_no, per_page=per_page))

available_models = [
    {"Name": "[CO] ResNetModel", "URL": "https://clarifai.com/clarifai/Streaming_module_inwork/models/detr-resnet-image-detection"},
    #{"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection"},
    #{"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection"},
    #{"Name": "Vehicle Detection", "URL": "https://clarifai.com/clarifai/main/models/vehicle-detector-alpha-x"},
]

selected_model_name = st.selectbox("Select a Model", [m["Name"] for m in available_models])
selected_model = next((m for m in available_models if m["Name"] == selected_model_name), None)
st.session_state.selected_model = selected_model["URL"] if selected_model else None

def get_bright_frame(video_url):
    cap = cv2.VideoCapture(video_url)
    frame = None
    for _ in range(10):
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()
    return frame
st.markdown("## 🎬 Select a Video")

thumbnails = []
video_urls = []
for inp in page_of_inputs:
    meta_stream_url = inp.data.metadata['stream_url']
    bright_frame = get_bright_frame(meta_stream_url)
    if bright_frame is not None:
        _, buffer = cv2.imencode(".jpg", bright_frame)
        img_bytes = BytesIO(buffer.tobytes())
        thumbnails.append(img_bytes)
        video_urls.append(meta_stream_url)

cols = st.columns(5)
for idx, img_src in enumerate(thumbnails):
    col = cols[idx % 5]
    with col:
        if img_src:
            st.image(img_src)
            if st.button(f"▶ Play Video {idx+1}", key=f"btn_{idx}"):
                st.session_state.selected_video = video_urls[idx]
                st.session_state.video_loaded = True
                st.rerun()

class AsyncVideoProcessor:
    def __init__(self, model_url, video_url, pat):
        if not model_url:
            raise ValueError("Model URL is required but missing.")
        self.model_url = model_url
        self.video_url = video_url
        self.pat = pat
        self.detector_model = Model(url=self.model_url, pat=self.pat)
        self.cap = cv2.VideoCapture(self.video_url)
        self.detections = {}
        self.frame_queue = deque(maxlen=80)  
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)

    def draw_detections(self, frame, frame_index):
        with self.lock:
            detections = self.detections.get(frame_index, [])
            for x1, y1, x2, y2, conc, value in detections:
                cv2.putText(frame, f"{conc} ({value:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def process_stream_batch(self):
        with self.lock:
            if not self.frame_queue:
                return
            batch = list(self.frame_queue)
            self.frame_queue.clear()
        proto_inputs = [
            Inputs.get_input_from_bytes(input_id=f'frame_{idx}', image_bytes=frame_bytes)
            for idx, (frame_bytes, frame_shape) in batch
        ]
        try:
            responses = self.detector_model.stream(iter([proto_inputs]))
            for idx, response in enumerate(responses):
                detections = []
                frame_shape = batch[idx][1] 
                for region in response.outputs[0].data.regions:
                    for concept in region.data.concepts:
                        conc = concept.name
                        value = concept.value
                    bbox = region.region_info.bounding_box
                    x1 = max(0, min(int(bbox.left_col * frame_shape[1]), frame_shape[1]))
                    y1 = max(0, min(int(bbox.top_row * frame_shape[0]), frame_shape[0]))
                    x2 = max(0, min(int(bbox.right_col * frame_shape[1]), frame_shape[1]))
                    y2 = max(0, min(int(bbox.bottom_row * frame_shape[0]), frame_shape[0]))
                    detections.append((x1, y1, x2, y2, conc, value))
                with self.lock:
                    self.detections[batch[idx][0]] = detections
        except Exception as e:
            print(f"Error in streaming inference: {e}")

    async def stream_video(self, frame_placeholder):
        frame_idx = 0
        input_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
            with self.lock:
                self.frame_queue.append((frame_idx, (frame_bytes, frame.shape)))
            if len(self.frame_queue) >= 8:
                self.executor.submit(self.process_stream_batch)
            frame = self.draw_detections(frame, frame_idx)
            _, buffer = cv2.imencode('.jpg', frame)
            img_bytes = BytesIO(buffer.tobytes())
            frame_placeholder.image(img_bytes, channels="RGB")
            await asyncio.sleep(1 / input_fps)
            frame_idx += 1
        self.cap.release()

if st.session_state.selected_video and st.session_state.selected_model:
    st.info("Streaming video with real-time inference...")
    processor = AsyncVideoProcessor(st.session_state.selected_model, st.session_state.selected_video, pat)
    frame_placeholder = st.empty()
    asyncio.run(processor.stream_video(frame_placeholder))

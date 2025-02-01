# Imports
import streamlit as st
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import asyncio
from io import BytesIO
from clarifai.client.auth.helper import ClarifaiAuthHelper
from clarifai.client.user import User
from clarifai.client.model import Model
from clarifai.client.input import Inputs
from clarifai.modules.css import ClarifaiStreamlitCSS
from clarifai_grpc.grpc.api.status import status_code_pb2
import threading

# Streamlit Config
st.set_page_config(layout="wide")
ClarifaiStreamlitCSS.insert_default_css(st)

# Setup Clarifai Authentication
auth = ClarifaiAuthHelper.from_streamlit(st)
pat = auth._pat

userDataObject = auth.get_user_app_id_proto()
user_id = userDataObject.user_id
app_id = userDataObject.app_id

input_obj = User(user_id=user_id).app(app_id=app_id, pat=pat).inputs()
page_no = 1
per_page = 25
page_of_inputs = list(input_obj.list_inputs(page_no=page_no, per_page=per_page))

# Configurations
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 8
executor = ThreadPoolExecutor(max_workers=10)
lock = threading.Lock()

# Dynamic frame skipping configuration
SKIP_THRESHOLD = 20  # If processed_frame_queue exceeds this, start skipping frames
skip_counter = 0  # Counter to control skipping

# Session State Initialization
if "selected_video" not in st.session_state:
  st.session_state.selected_video = None
if "selected_model" not in st.session_state:
  st.session_state.selected_model = None

# Function to get a bright frame for thumbnails
def get_bright_frame(video_url, brightness_threshold=50, max_attempts=25):
  cap = cv2.VideoCapture(video_url)
  frame = None

  for _ in range(max_attempts):
    ret, frame = cap.read()
    if not ret:
      break

    # Convert frame to grayscale to measure brightness
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray_frame)

    # Check if frame brightness exceeds the threshold
    if brightness > brightness_threshold:
      break  # Found a sufficiently bright frame

  cap.release()
  return frame

# Video Processing Class
class VideoProcessor:
  def __init__(self, model_url, video_url, pat):
    self.model_url = model_url
    self.video_url = video_url
    self.pat = pat
    self.detector_model = Model(url=self.model_url, pat=self.pat)
    self.cap = cv2.VideoCapture(self.video_url)
    self.frame_queue = []
    self.processed_frame_queue = []
    self.frame_counter = 0
    self.min_buffer_size = BATCH_SIZE * 2  # Ensure at least 2 batches before playback starts

  def prepare_frame_for_batch(self, frame):
    self.frame_counter += 1
    input_id = f'frame_{self.frame_counter}'
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_bytes = cv2.imencode('.jpg', frame_rgb)[1].tobytes()
    proto_input = Inputs.get_input_from_bytes(input_id=input_id, image_bytes=frame_bytes)
    return proto_input, self.frame_counter

  def draw_predictions_on_frame(self, frame, prediction_response, frame_num):
    cv2.putText(frame, f"Frame: {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    if prediction_response.status.code != status_code_pb2.SUCCESS:
      return frame

    if hasattr(prediction_response, "data") and hasattr(prediction_response.data, "regions"):
      for region in prediction_response.data.regions:
        if region.value < CONFIDENCE_THRESHOLD:
          continue

        bbox = region.region_info.bounding_box
        top = int(bbox.top_row * frame.shape[0])
        left = int(bbox.left_col * frame.shape[1])
        bottom = int(bbox.bottom_row * frame.shape[0])
        right = int(bbox.right_col * frame.shape[1])

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        for concept in region.data.concepts:
          name = concept.name
          value = round(concept.value, 4)
          label = f"{name}: {value}"
          cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

  def batch_process_frames(self, frames_with_numbers):
    proto_list = [proto for proto, _ in frames_with_numbers]
    batch_prediction = self.detector_model.predict(proto_list)
    return list(zip(batch_prediction.outputs, [num for _, num in frames_with_numbers]))

  async def display_processed_frames(self, frame_placeholder, frame_interval):
    while len(self.processed_frame_queue) < self.min_buffer_size:
      await asyncio.sleep(0.1)

    while True:
      if self.processed_frame_queue:
        with lock:
          processed_frame = self.processed_frame_queue.pop(0)
        _, buffer = cv2.imencode('.jpg', processed_frame)
        img_bytes = BytesIO(buffer.tobytes())
        frame_placeholder.image(img_bytes, channels="RGB")
        await asyncio.sleep(frame_interval)
      else:
        await asyncio.sleep(0.01)

  async def process_video_async(self, frame_placeholder):
    global skip_counter
    loop = asyncio.get_event_loop()
    input_fps = self.cap.get(cv2.CAP_PROP_FPS)
    frame_interval = 1.0 / input_fps if input_fps > 0 else 0.03

    display_task = asyncio.create_task(self.display_processed_frames(frame_placeholder, frame_interval))

    while self.cap.isOpened():
      ret, frame = self.cap.read()
      if not ret:
        break

      # Dynamic frame skipping logic
      with lock:
        if len(self.processed_frame_queue) > SKIP_THRESHOLD:
          skip_counter += 1
          if skip_counter % 2 == 0:  # Skip every other frame if buffer is too large
            continue

      proto_input, frame_num = self.prepare_frame_for_batch(frame)
      self.frame_queue.append((proto_input, frame, frame_num))

      if len(self.frame_queue) >= BATCH_SIZE:
        batch_frames = self.frame_queue[:BATCH_SIZE]
        self.frame_queue = self.frame_queue[BATCH_SIZE:]

        batch_outputs_with_numbers = await loop.run_in_executor(executor, self.batch_process_frames, [(proto, num) for proto, _, num in batch_frames])

        for idx, (output, frame_num) in enumerate(batch_outputs_with_numbers):
          processed_frame = self.draw_predictions_on_frame(batch_frames[idx][1], output, frame_num)
          with lock:
            self.processed_frame_queue.append(processed_frame)

    self.cap.release()
    await display_task

# Display Thumbnails & Model Selection
st.markdown("## 🎬 Select a Video & Model")

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
  else:
    thumbnails.append(None)
    video_urls.append(None)

while len(thumbnails) % 5 != 0:
  thumbnails.append(None)
  video_urls.append(None)

cols = st.columns(5)
for idx, img_src in enumerate(thumbnails):
  col = cols[idx % 5]
  with col:
    if img_src:
      st.image(img_src)
      if video_urls[idx]:
        if st.button("▶ Play", key=f"btn_{idx}"):
          st.session_state.selected_video = video_urls[idx]
          st.session_state.video_loaded = True
          st.rerun()
    else:
      st.write("")

# Model Selection Dropdown
st.markdown("## AI Processing Settings")
available_models = [
  {"Name": "General-Image-Detection", "URL": "https://clarifai.com/clarifai/main/models/general-image-detection"},
  {"Name": "Face Detection", "URL": "https://clarifai.com/clarifai/main/models/face-detection"},
  {"Name": "Vehicle Detection", "URL": "https://clarifai.com/clarifai/main/models/vehicle-detector-alpha-x"}
]

model_names = [model["Name"] for model in available_models]
selected_model_name = st.selectbox("Select a Model", model_names)
selected_model = next(model for model in available_models if model["Name"] == selected_model_name)
st.session_state.selected_model = selected_model["URL"]

# Real-Time Processed Video Display
st.markdown("## 🔄 Live Model Inference")

if st.session_state.selected_video and st.session_state.selected_model:
  st.info("Model is processing the video...")

  processor = VideoProcessor(st.session_state.selected_model, st.session_state.selected_video, pat)
  frame_placeholder = st.empty()

  asyncio.run(processor.process_video_async(frame_placeholder))

else:
  st.warning("Please select a video and a model to start processing.")
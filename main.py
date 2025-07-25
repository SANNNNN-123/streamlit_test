"""Basic webcam demo with Streamlit WebRTC."""

import logging
from pathlib import Path
from typing import List

import av
import numpy as np
import streamlit as st
from streamlit_webrtc import (
    WebRtcMode,
    webrtc_streamer,
    __version__ as st_webrtc_version,
)
import aiortc

HERE = Path(__file__).parent
ROOT = HERE

logger = logging.getLogger(__name__)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Basic video frame callback that just returns the frame as is."""
    image = frame.to_ndarray(format="bgr24")
    
    # You can add basic image processing here if needed
    # For now, just return the original frame
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")


webrtc_ctx = webrtc_streamer(
    key="webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    video_frame_callback=video_frame_callback,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.markdown("## Basic Webcam Demo")
st.markdown("This is a simple webcam streaming demo using Streamlit WebRTC.")

if webrtc_ctx.state.playing:
    st.success("Webcam is active!")

st.markdown(
    f"Streamlit version: {st.__version__}  \n"
    f"Streamlit-WebRTC version: {st_webrtc_version}  \n"
    f"aiortc version: {aiortc.__version__}  \n"
) 
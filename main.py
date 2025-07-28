"""
Basic webcam demo with Streamlit WebRTC.

To run this, you need to have a Twilio account and 
set the TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN .env

https://github.com/whitphx/streamlit-webrtc#configure-the-turn-server-if-necessary
"""


import logging
from pathlib import Path
from typing import List
import os
from twilio.rest import Client

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

account_sid = os.environ['TWILIO_ACCOUNT_SID']
auth_token = os.environ['TWILIO_AUTH_TOKEN']
client = Client(account_sid, auth_token)

token = client.tokens.create()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    """Basic video frame callback that just returns the frame as is."""
    image = frame.to_ndarray(format="bgr24")
    
    # You can add basic image processing here if needed
    # For now, just return the original frame
    
    return av.VideoFrame.from_ndarray(image, format="bgr24")

#rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},

webrtc_ctx = webrtc_streamer(
    key="webcam",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": token.ice_servers},
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

    """
    ctx = webrtc_streamer(key="stocazzo", video_frame_callback=video_frame_callback)
    
    while ctx.state.playing:
        with lock:
            img = img_container["img"]
            if img is None:
                continue
            ss.sframe_list.append(img)
        print(ss.sframe_list[-1])
        
        
    print("shape of the list", len(ss.sframe_list))
    print(np.array(ss.sframe_list).shape)
    st.video(np.array(ss.sframe_list))

    lock = threading.Lock()
    img_container = {"img": None}
    img = torch.zeros(1,1)
    def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img
    return frame
    """
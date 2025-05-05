import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from 'react';

const WebcamViewer = forwardRef(({ onCapture }, ref) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [isPaused, setIsPaused] = useState(false);

  // Refs for real-time access inside async loops
  const isPausedRef = useRef(false);
  const isProcessingRef = useRef(false);

  //starts camera
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;

        // Wait for the video to load and then call sendFrame
        videoRef.current.onloadeddata = () => {
          if (!isPausedRef.current) sendFrame();
        };

      } catch (err) {
        console.error('Error accessing webcam:', err);
      }
    };

    startCamera();
    
    return () => {
      const stream = videoRef.current?.srcObject;
      stream?.getTracks().forEach(track => track.stop());
    };
  }, []);

  // Keep refs in sync with state
  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);


  const captureFrame = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    return new Promise((resolve) => {
      canvas.toBlob(blob => {
        if (blob) {
          resolve(blob);
        }
      }, 'image/jpeg', 1);
    });
  };

  const sendFrame = async () => {
    console.log('sendframe')
    if (isPausedRef.current || isProcessingRef.current) return;

    isProcessingRef.current = true;
    const blob = await captureFrame();
    if (!blob) {
      isProcessingRef.current = false;
      return;
    }

    const formData = new FormData();
    formData.append('image', blob, 'frame.jpg');

    try {
      const response = await fetch('https://f910-2600-1017-a410-36b8-2357-52be-1318-959b.ngrok-free.app/process/', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      onCapture?.(data); //send to parent
    } catch (err) {
      console.error('Error sending frame:', err);
    } finally {
      isProcessingRef.current = false;
      if (!isPausedRef.current) {
        sendFrame();
      }
    }
  }

  const togglePause = () => {
    const stream = videoRef.current?.srcObject;
    if (!stream) return;

    const track = stream.getVideoTracks()[0];
    track.enabled = isPausedRef.current;

    const newPausedState = !isPausedRef.current;
    setIsPaused(newPausedState);
    isPausedRef.current = newPausedState;

    if (!newPausedState && !isProcessingRef.current) {
      sendFrame();
    }
  };

 // Expose capture function to parent
useImperativeHandle(ref, () => ({
  captureNow: sendFrame,
}));

  return (
    <div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
      />
      <button onClick={togglePause} style={{ fontSize: "16px", marginTop: '20px', padding: '10px 15px', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '5px' }}>
        {isPaused ? 'Resume' : 'Pause'}
      </button>

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
});

export default WebcamViewer;
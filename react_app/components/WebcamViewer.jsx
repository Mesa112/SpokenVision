import React, { useEffect, useRef, useState, useImperativeHandle, forwardRef } from 'react';

const WebcamViewer = forwardRef(({ onCapture }, ref) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);

  //starts camera
  useEffect(() => {
    const startCamera = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({ video: true });
        videoRef.current.srcObject = stream;

        // Wait for the video to load and then call sendFrame
        videoRef.current.onloadeddata = () => {
          if (onCapture) onCapture(); // signal parent video is ready
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

 // Expose capture function to parent
useImperativeHandle(ref, () => ({
  captureNow: captureFrame,
  pauseStream: () => {
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getVideoTracks().forEach(track => track.enabled = false);
    }
  },
  resumeStream: () => {
    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getVideoTracks().forEach(track => track.enabled = true);
    }
  }
}));

  return (
    <div>
      <video
        ref={videoRef}
        autoPlay
        playsInline
        muted
      />

      <canvas ref={canvasRef} style={{ display: 'none' }} />
    </div>
  );
});

export default WebcamViewer;
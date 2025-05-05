
import { useState, useEffect, useRef } from 'react'
import WebcamViewer from '../components/webcamViewer.jsx'
import './App.css';

function App() {
  //create react state variables image and message
  const [image, setImage] = useState(null);
  const [message, setMessage] = useState("");
  const [captions, setCaptions] = useState([]);
  const [currentAudio, setCurrentAudio] = useState(null); 

  const webcamRef = useRef(null);
  const [isPaused, setIsPaused] = useState(false);
  const isProcessingRef = useRef(false); 
  const isPausedRef = useRef(isPaused); //for tracking in async functions


  const sendFrame = async () => {
    console.log("SENDING")
    if (isPaused || isProcessingRef.current) return;
  
    isProcessingRef.current = true;
  
    const blob = await webcamRef.current.captureNow();
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
      console.log("Response from server:", data);
      handleResponse(data);
    } catch (err) {
      console.error('Error sending frame:', err);
    } finally {
      isProcessingRef.current = false;
    }
  };

  useEffect(() => {
    isPausedRef.current = isPaused;
  }, [isPaused]);

  //runs every second to send frames to server unless paused or already waiting for a response
  useEffect(() => {
    if (isPaused) return;

    const intervalId = setInterval(() => {
      if (!isProcessingRef.current) {
        sendFrame();
      }
    }, 1000); // 1 frame per second.
  
    return () => clearInterval(intervalId);
  }, [isPaused]);

  // Called once when video is loaded to start sending frames
  const handleCameraReady = () => {
    setIsPaused(false);
    if (!isPaused) {
      sendFrame();
    }
  };

  const togglePause = () => {
    setIsPaused(prev => {
      const newPaused = !prev;
      
      if (newPaused) {
        webcamRef.current.pauseStream(); // pause video
        stopAudio();
      } else {
        webcamRef.current.resumeStream(); // resume video
      }
  
      return newPaused;
    });
  };

  //runs when user selects a file
  const handleFileChange = (event) => {
    setImage(event.target.files[0]) //just one
  };

  //runs when user submits file
  const handleSubmit = async (event) => {
    event.preventDefault() //no refresh
    if (!image || !image.type.startsWith('image/')) { //checks to see if an image type was uploaded
      alert("That's not an image")
      return
    }
   
    const uploadData = new FormData()  //
    uploadData.append('image', image) //create JS object, then send to backend shown below VV

    try {
      const response = await fetch('http://localhost:5000/upload', {
        method: 'POST',
        body: uploadData
      }); 

      const imageDescription = await response.json()
      setMessage(imageDescription.message)

      // try to send image to server, then wait for response and parse into json

      // Play audio when the server returns audio
      // const audio = new Audio(imageDescription.audioUrl)
      // audio.play()

    } catch (error) {
      console.error('Upload failed:', error)
      setMessage("Failed to process image.")
    };
  }

  const stopAudio = () => {
    if (currentAudio) {
      currentAudio.pause(); // Pause the currently playing audio
      currentAudio.currentTime = 0; // Reset playback to the beginning
      setCurrentAudio(null); // Clear the current audio reference
    }
  }
  const playAudio = (audio) => {
    stopAudio()
    audio.play().catch(err => console.warn("Autoplay failed:", err));
    setCurrentAudio(audio); // Set the current audio to the new one
  };

  //handles response from server
  //plays audio and appends caption to the list of captions UNLESS paused
  const handleResponse = async (data) => {
    if (isPausedRef.current) {
      console.log("Ignored response, video is paused.");
      return;
    }
    //auto play audio
    const audio = new Audio(`data:audio/wav;base64,${data.audio_base64}`);
    playAudio(audio) //autoplay audio from server

    const timestamp = new Date().toLocaleTimeString(); //get current time
    setCaptions(prev => [{ caption: data.caption, audio, timestamp }, ...prev]); //appends new caption to the list of captions
  };

  return (
    
    //Tailwind template from creative-tim.com
    <div className="h-screen flex">
    {/*------------------------ LEFT SIDE: background + heading ---------------------------------*/}
    <div
      className="hidden lg:flex w-full lg:w-1/3 justify-around items-center bg-cover bg-center"
      style={{
        backgroundImage: `linear-gradient(rgba(2,2,2,0.7), rgba(0,0,0,0.7)), url('https://images.unsplash.com/photo-1650825556125-060e52d40bd0?ixlib=rb-1.2.1&ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&auto=format&fit=crop&w=1170&q=80')`,
      }}
    >
      <div className="text-center text-white space-y-6 px-12">
        <h1 className="text-4xl font-bold">Clear View MVP</h1>
        <p className="text-white text-lg">Upload and listen to your environment</p>
      </div>
    </div>
    
    {/* ---------------------------RIGHT SIDE: upload form ----------------------------------------*/}

    <div className="flex flex-col w-full lg:w-2/3 justify-center items-center bg-gray-900" id = "rightContainer">
      <h1 className="text-4xl font-bold text-white mb-6">Clear View MVP</h1>
      {/* Webcam component */}
      <WebcamViewer ref={webcamRef} onCapture={handleCameraReady} />
      <button
        onClick={togglePause}
        style={{ fontSize: "16px", marginTop: '20px', padding: '10px 15px', backgroundColor: '#007BFF', color: 'white', border: 'none', borderRadius: '5px' }}
      >
        {isPaused ? 'Resume' : 'Pause'}
      </button>
      {/* <form onSubmit={handleSubmit} className="bg-gray-800 text-white rounded-lg shadow-2xl p-8 w-full max-w-md space-y-6">
        <h2 className="text-xl font-semibold text-center text-blue-400">Upload an Image</h2>
        <input
          type="file"
          accept="image/*"
          onChange={handleFileChange}
          className="w-full bg-gray-700 text-white border border-gray-600 rounded px-3 py-2 text-sm"
          />
        <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded transition">
          Submit 
        </button>
        {message && (
          <div className="mt-4 text-red-400 text-sm text-center">
            <strong>Feedback:</strong> {message}
          </div>
        )}
      </form> */}

      {/* Captions display */}
      <div id="captionsContainer">
        {captions.map((item, index) => (
          <div key={index} style={{
            padding: '10px',
            marginBottom: '10px',
            border: '1px solid #ccc',
            borderRadius: '5px',
            background: '#f9f9f9'
          }}>
            <p><strong>{item.timestamp}</strong></p>
            <p>{item.caption}</p>
            <button
              onClick={() => playAudio(item.audio)} // Call playAudio with the specific audio object
              style={{ 
                fontSize: "16px", 
                marginTop: '20px', 
                padding: '10px 15px', 
                backgroundColor: '#007BFF', 
                color: 'white', 
                border: 'none', 
                borderRadius: '5px'
              }}
            >
              Play Audio
            </button>
          </div>
        ))}
      </div>


    </div>
  </div>
);
}

export default App

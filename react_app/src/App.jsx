
import { useState } from 'react'


function App() {

  //create react state variables image and message
  const [image, setImage] = useState(null);
  const [message, setMessage] = useState("");


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

  return (
    
    //Tailwind template from creative-tim.com
    <div className="h-screen flex">

    {/*------------------------ LEFT SIDE: background + heading ---------------------------------*/}

    <div
      className="hidden lg:flex w-full lg:w-1/2 justify-around items-center bg-cover bg-center"
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

    <div className="flex w-full lg:w-1/2 justify-center items-center bg-gray-900">

      <form onSubmit={handleSubmit} className="bg-gray-800 text-white rounded-lg shadow-2xl p-8 w-full max-w-md space-y-6">

              <h2 className="text-xl font-semibold text-center text-blue-400">Upload an Image</h2>

              <input
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full bg-gray-700 text-white border border-gray-600 rounded px-3 py-2 text-sm"
                />

              <button type="submit" className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 rounded transition">

                Submit {/* Submit */ }

              </button>

              {message && (

                <div className="mt-4 text-red-400 text-sm text-center">
                  <strong>Feedback:</strong> {message}
                </div>
                
              )}

      </form>

    </div>

  </div>

);
  

}

export default App

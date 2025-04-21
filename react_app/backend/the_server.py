


#python the_server.py to run
#make sure there is an uploads/ folder before running

from flask import Flask, request, jsonify
from flask_cors import CORS
import os



the_server = Flask(__name__)
CORS(the_server)  # allow React frontend to make requests


UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads") #get to the upload folder


@the_server.route('/upload', methods=['POST'])

#not sure how you are supposed to get ML into this response
def upload_file():

    file = request.files.get('image')

    if file and file.filename.lower().endswith(('png', 'jpg', 'jpeg','gif','cr2')):#cam? 

        filepath = os.path.join(UPLOAD_FOLDER, file.filename) #save the file name
        file.save(filepath) #then the content

      
        message = f"Image '{file.filename}' uploaded and processed."

        return jsonify({

            "message": message,
            #text description "http://localhost:5000/textdescrip.something
            #"audioUrl": "http://localhost:5000/audio.something

        })
    

    return jsonify({ "message": "Invalid file." }), 400 #http



if __name__ == '__main__':

    the_server.run(port=5000, debug=True)

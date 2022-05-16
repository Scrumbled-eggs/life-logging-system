from flask import Flask, jsonify, request
from flask_restx import Api

app = Flask(__name__)
api = Api(app)

@app.route('/logs/<userid>')   
def modelfr(userid):    
    print(f'request!{userid}')
    if request.method == 'GET':
        return 'drink water, drink water'

if __name__ == '__main__':
    app.run(host="0.0.0.0", debug=False) 
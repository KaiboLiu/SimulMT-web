from flask import stream_with_context, request, Response

app = flask.Flask(__name__)


@app.route('/stream')
def streamed_response():
    def generate():
        yield 'Hello '
        yield request.args['name']
        yield '!'
    return Response(stream_with_context(generate()))

if __name__ == "__main__":
    app.run(debug=True)

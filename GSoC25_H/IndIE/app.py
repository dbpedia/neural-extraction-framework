from flask import Flask, render_template, request, flash
from chunck import ChunckProcessor
import os

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY',"SECRET_KEY")
processor = {
    "hi": ChunckProcessor(language="hi"),
    "ta": ChunckProcessor(language="ta"),
    "te": ChunckProcessor(language="te"),
    "ur": ChunckProcessor(language="ur")
}


@app.route('/', methods=('GET', 'POST'))
def create():
    if request.method == 'POST':
        sentence = request.form['sentence']
        language = request.form['language']
        if not sentence or not language:
            flash('Sentence and Language are required!')
        else:
            lang_processor = processor.get(language)
            data = []
            if lang_processor:
                result = lang_processor.run(sentence)
                for sentence, triple in zip(result[0], result[1]):
                    obj = {
                        "sentence": sentence,
                        "result": triple
                    }
                    data.append(obj)
            return render_template('index.html', data=data)

    return render_template('index.html', data=[])
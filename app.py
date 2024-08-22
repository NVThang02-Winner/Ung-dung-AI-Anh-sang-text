from flask import Flask, request, render_template
import requests
from PIL import Image
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from functools import lru_cache

app = Flask(__name__)

# Sử dụng cache để lưu trữ kết quả của các yêu cầu đã xử lý
@lru_cache(maxsize=128)
def get_image_answer(img_url, question):
    response = requests.get(img_url)
    image = Image.open(BytesIO(response.content)).convert('RGB')

    # Giảm kích thước hình ảnh để tăng tốc độ xử lý
    # Bạn có thể điều chỉnh kích thước này tùy thuộc vào yêu cầu cụ thể
    image = image.resize((256, 256))

    inputs = processor(image, question, return_tensors="pt")
    outputs = model.generate(**inputs)
    answer = processor.decode(outputs[0], skip_special_tokens=True)
    return answer

# Khởi tạo mô hình và processor
processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xl")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-flan-t5-xl")

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        img_url = request.form['img_url']
        question = request.form['question']

        # Lấy câu trả lời từ mô hình, sử dụng cache để giảm thời gian xử lý
        answer = get_image_answer(img_url, question)
        
        return render_template('index.html', answer=answer, img_url=img_url)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

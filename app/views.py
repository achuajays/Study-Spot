import os
from dotenv import load_dotenv
from openai import OpenAI
from django.shortcuts import render , redirect
from django.contrib.auth.models import User
from .models import imagess , AudioFiles , TextFiles , UserResponse , Content , UploadedFiles , Marks , MP4Details
from .PdfProcessor import PdfProcessor
from .ExtractContent import ImageOCR
from markdownify import markdownify as md
import google.generativeai as genai


pdf = PdfProcessor()
# Use a pipeline as a high-level helper


load_dotenv()  # take environment variables from .env.
api_key = os.getenv('API_KEY')  # external config variable in Django
api_key_gem = os.getenv('api_key_gem')
client = OpenAI(
    api_key = api_key
)
genai.configure(api_key=api_key_gem)
# Create your views here.
def home(request):
    category = request.user.first_name
    print(category)
    return render(request, 'app/home.html', {'category': category})





def logic(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a logical assistant, skilled in explaining any program in a logical manner."},
                    {"role": "user", "content": user_input}
                ]
            )

            output = completion.choices[0].message.content
            output  =  md(output)
            print(output)
            return render(request, 'app/logic.html', {'output': output})
    return render(request, 'app/logic.html', {})


def law(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a law assistant, skilled in explaining any  indian law in a logical manner with proper examplee . from input problem (smaller point)"},
                    {"role": "user", "content": user_input}
                ]
            )

            output = completion.choices[0].message.content
            return render(request, 'app/law.html', {'output': output})
    return render(request, 'app/law.html', {})




def translator(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a translator assistant, skilled in translating to english "},
                    {"role": "user", "content": user_input}
                ]
                )

            output = completion.choices[0].message.content
            return render(request, 'app/translater.html', {'output': output})
    return render(request, 'app/translater.html', {})


import io
import torch
from PIL import Image
import torchvision.transforms as transforms
from django.http import HttpResponse
from django.shortcuts import render

norm_layer = torch.nn.InstanceNorm2d

class ResidualBlock(torch.nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features),
                      torch.nn.ReLU(inplace=True),
                      torch.nn.ReflectionPad2d(1),
                      torch.nn.Conv2d(in_features, in_features, 3),
                      norm_layer(in_features)]

        self.conv_block = torch.nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)


class Generator(torch.nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9, sigmoid=True):
        super(Generator, self).__init__()

        # Initial convolution block
        model0 = [torch.nn.ReflectionPad2d(3),
                  torch.nn.Conv2d(input_nc, 64, 7),
                  norm_layer(64),
                  torch.nn.ReLU(inplace=True)]
        self.model0 = torch.nn.Sequential(*model0)

        # Downsampling
        model1 = []
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model1 += [torch.nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                       norm_layer(out_features),
                       torch.nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features * 2
        self.model1 = torch.nn.Sequential(*model1)

        model2 = []
        # Residual blocks
        for _ in range(n_residual_blocks):
            model2 += [ResidualBlock(in_features)]
        self.model2 = torch.nn.Sequential(*model2)

        # Upsampling
        model3 = []
        out_features = in_features // 2
        for _ in range(2):
            model3 += [torch.nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                       norm_layer(out_features),
                       torch.nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features // 2
        self.model3 = torch.nn.Sequential(*model3)

        # Output layer
        model4 = [torch.nn.ReflectionPad2d(3),
                  torch.nn.Conv2d(64, output_nc, 7)]
        if sigmoid:
            model4 += [torch.nn.Sigmoid()]

        self.model4 = torch.nn.Sequential(*model4)

    def forward(self, x, cond=None):
        out = self.model0(x)
        out = self.model1(out)
        out = self.model2(out)
        out = self.model3(out)
        out = self.model4(out)

        return out

# Load models
model1 = Generator(3, 1, 3)
model1.load_state_dict(torch.load(r'C:\Users\aotir\Downloads\Desktop\Project\jyothi_hackathena\app\model.pth', map_location=torch.device('cpu')))
model1.eval()

model2 = Generator(3, 1, 3)
model2.load_state_dict(torch.load(r'C:\Users\aotir\Downloads\Desktop\Project\jyothi_hackathena\app\model2.pth', map_location=torch.device('cpu')))
model2.eval()

from django.core.files.base import ContentFile
import pathlib
import textwrap



def predictt(request):
    if request.method == 'POST':
        # Get input image and version from the request
        input_img = request.FILES['file']
        ver = request.POST.get('version')

        # Load the input image
        input_image = Image.open(input_img)
    
        

        # Transform the input image
        transform = transforms.Compose([transforms.Resize(256, Image.BICUBIC), transforms.ToTensor()])
        input_image = transform(input_image)
        input_image = torch.unsqueeze(input_image, 0)

        # Perform prediction based on the selected version
        with torch.no_grad():
            if ver == 'Simple Lines':
                drawing = model2(input_image)[0].detach()
            else:
                drawing = model1(input_image)[0].detach()

        # Convert the output to a PIL image
        drawing = transforms.ToPILImage()(drawing)

        # Save the drawing image to the database
        drawing_bytes = io.BytesIO()
        drawing.save(drawing_bytes, format='PNG')
        drawing_bytes = drawing_bytes.getvalue()

        # Create a new ImageImage object and save the image data
        drawing_instance = imagess(image=ContentFile(drawing_bytes, name='drawing.png'))
        drawing_instance.save()
        print(drawing_instance.image.url)
        

        # Return a response or redirect as needed
        return render(request, 'app/l.html' , {'n' : drawing_instance })
    
        # Handle GET requests (if necessary)
    return render(request, 'app/q.html')


import io
from django.shortcuts import render
# Import your API client here

import os
import os
import io
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from openai import OpenAI


# def transcribe_audio(request):
#     if request.method == 'POST':
#         audio_file = request.FILES['audio_file']

#         # Save the uploaded audio file
#         fs = FileSystemStorage()
#         file_path = fs.save(audio_file.name, audio_file)  # Get the saved file path

#         # Authenticate with OpenAI API using your API key
        

#         # Read the saved file and send to OpenAI API for transcription
#         with io.open(file_path, 'rb') as f:
#             transcription = client.audio.transcriptions.create(
#                 model="whisper-1", 
#                 file=f, 
#                 response_format="text"
#             )
#         return render(request, 'app/audio_to_text.html', {'output': transcription.text})
#     return render(request, 'app/audio.html')




def transcribe_audio(request):
    if request.method == 'POST':
        audio_file = request.FILES['audio_file']
        # Save the uploaded audio file to the AudioFiles database
        audio_instance = AudioFiles(audio_file=audio_file)
        audio_instance.save()

        # Read the saved file and send to OpenAI API for transcription
        with io.open(audio_instance.audio_file.path, 'rb') as f:
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="text"
            )
            if transcription:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a note building assistant, skilled in creating large text to  easy to learn bullet point with good understabding always a bullet point only can have 20 charecter "},
                        {"role": "user", "content": transcription}
                    ]
                )

                output = completion.choices[0].message.content

                # Save the output to a TextFile database
                text_instance = TextFiles(text_file=ContentFile(output.encode(), name='output.txt'))
                text_instance.save()
                # Return the download URL in the render
                return render(request, 'app/audio_to_text.html', {'output': output, 'download_url': text_instance.text_file.url})
                # Save the output to a TextFile database
                

    return render(request, 'app/audio.html')

from django.shortcuts import render
from django.http import HttpResponse
import torch
from .image_captioning import predicted  # Assuming your image captioning function is in image_captioning.py
from .caption import generater

import os
from django.conf import settings

def caption_image(request):
    if request.method == 'POST':
        image_file = request.FILES.get('image')
        if image_file:
            # Save the uploaded image to a file
            image_path = os.path.join(settings.MEDIA_ROOT, 'uploaded_image.jpg')
            with open(image_path, 'wb') as f:
                for chunk in image_file.chunks():
                    f.write(chunk)

            # Save the uploaded image to Images database
            image = imagess(image=ContentFile(image_file.read(), name='uploaded_image.jpg'))
            image.save()

            # Call predict function with the file path
            caption = predicted(image_path)

            # Optionally, delete the uploaded image file after processing
            # os.remove(image_path)
            caption = remove_text(caption, '<|endoftext|>')
            if caption:
                completion = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a caption assistant, skilled in creating large explanation  where each line is max 20 charecter and min 2 line and 2 line max 10 line and given a simple sentence with only caption "},
                        {"role": "user", "content": caption}
                    ]
                )

                output = completion.choices[0].message.content

            return render(request, 'app/caption.html', {'caption': caption,  'output' : output })
    return render(request, 'app/caption.html')


def remove_text(original_text, text_to_remove):
    # Replace the text to remove with an empty string
    modified_text = original_text.replace(text_to_remove, "")
    return modified_text



from django.shortcuts import render
from django.http import HttpResponse
from .models import UserResponse

def carrier(request):
    return render(request, 'app/carrier.html')

def process_responses(request):
    if request.method == 'POST':
        responses = []
        for field_name in [
            "activity",
            "ideal_workday",
            "subject",
            "industry",
            "skill",
            "values",
            "past_experience",
            "team_or_individual",
            "creativity_or_analysis",
            "career_goals"
        ]:
            value = request.POST.get(field_name)
            if value and value.strip():
                responses.append(value.strip())

        
        # Assuming you have a function to calculate accuracy based on responses
        output = calculate_accuracy(responses)
        
        # Save user responses
        user_response = UserResponse(response=output)
        user_response.save()
        # Save output to a new txt file
        text_instance = TextFiles(text_file=ContentFile(output.encode(), 'output.txt'))
        text_instance.save()
        download_url = text_instance.text_file.url

        
        return render(request, 'app/result.html', {'id': user_response.id,'output': output , 'download_url' : download_url ,})
    else:
        return HttpResponse('Method not allowed')
    

def calculate_accuracy(responses):
    # Assuming you have a function to calculate accuracy based on responses
    if responses:
            value = """act a good carrier counsiler for  student What activities or hobbies do you find yourself naturally gravitating towards in your free time?
When you envision your ideal workday, what tasks or responsibilities are you performing?
What subjects or topics do you enjoy learning about the most?
Is there a particular industry or field that has always piqued your interest?
Are there any specific skills or talents you possess that you're particularly proud of?
What are your core values, and how do you envision them aligning with your future career?
Have you had any past experiences (e.g., internships, part-time jobs, volunteer work) that have influenced your career interests?
Do you prefer working independently or as part of a team?
Are you more drawn to roles that involve creativity and innovation, or those that require analytical thinking and problem-solving?
What are your long-term career goals, and what steps do you think you need to take to achieve them? cosider them with list of answer and display and prepare a txt prepare me a report with 2 thing my job - matching persentage , why  """
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a carrier counsiler for  student "},
                    {"role": "user", "content": f"question - {value} and answer - {responses}create a detailed report"}
                ]
            )

            output = completion.choices[0].message.content

    return output 


from django.http import HttpResponse
from django.shortcuts import redirect
from django.core.files.base import ContentFile
import tempfile
import os
from openai import Client

def download_resume_template(request, id = id):
    # Initialize OpenAI client
    l = UserResponse.objects.get(id = id)
    try:
        # Generate resume suggestions
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a resume counselor for a student."},
                {"role": "user", "content": f"create a resume suggestion for {l.response} for freasher and area to focus and main skills and details of student carrier prefference"} 
            ]
        )

        # Extract output from completion
        output_content = completion.choices[0].message.content

        # Save output to a temporary file
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            temp_file.write(output_content)
            temp_file_path = temp_file.name

        # Create a file-like response for downloading
        with open(temp_file_path, 'rb') as file:
            response = HttpResponse(file.read(), content_type='text/plain')
            response['Content-Disposition'] = 'attachment; filename=output.txt'

        # Delete the temporary file
        os.unlink(temp_file_path)

        return response

    except Exception as e:
        # Handle errors appropriately
        return HttpResponse("An error occurred: {}".format(e), status=500)
    




from django.http import HttpResponse
from django.shortcuts import render
from pptx import Presentation
import openai
from io import BytesIO
# Set up OpenAI API key




def register(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        category = request.POST.get('category')
        if username and password:
            user = User.objects.create_user(username=username, password=password , first_name = category)
            user.save()
            return redirect ( register )
    else:
        return render(request, 'app/register.html')
    

def save_text_file(request):
    if request.method == 'POST':
        text_file = request.FILES.get('text_file')
        if text_file:
            # Save the uploaded text file to the TextFiles database
            text_instance = TextFiles(text_file=text_file)
            text_instance.save()

            # Convert PDF to images and extract text
            image_path = PdfProcessor.pdf_to_images(text_instance.text_file.path, text_instance.text_file.name)
            content = ''
            for file in image_path:
                text_extractor = ImageOCR()
                extracted_content = text_extractor.extract_text(file)
                content += ' ' + extracted_content

            # Save the extracted content to the Content model
            Content(name=text_instance.text_file.name, content=content).save()

            return render(request, 'app/upload_text.html', {'success': 'File uploaded successfully'})

    return render(request, 'app/upload_text.html')


from django.shortcuts import render, redirect
from .models import TextFiles
import PyPDF2
import warnings





def extract_text_from_pdf(pdf_file):
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    text = ""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def upload_pdf(request):
    if request.method == 'POST':
        pdf_file = request.FILES.get('pdf_file')
        if pdf_file:
            extracted_text = extract_text_from_pdf(pdf_file)
            instance = Content(name=pdf_file, content=extracted_text)
            instance.save()
            return redirect(extracted_text_detail , id=instance.id)
    return render(request, 'app/upload_pdf.html')

def extracted_text_detail(request, id = id ):
    extracted_text = Content.objects.get(id = id)
    question = request.POST.get('question')
    if question:
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a question answering expert , you can answer any question from a paragraph content  for a student."},
                {"role": "user", "content": f"content - {extracted_text.content} \n question - {question}"} 
            ]
        )

        # Extract output from completion
        output_content = completion.choices[0].message.content
        
        return render(request, 'app/extracted_text_detail.html', {'extracted_text': extracted_text , 'answer': output_content , 'question': question})
    return render(request, 'app/extracted_text_detail.html', {'extracted_text': extracted_text})


# views.py
import os
from django.shortcuts import render
from django.http import HttpResponse
from PyPDF2 import PdfReader
from paddleocr import PaddleOCR

def ats(request):
    if request.method == 'POST' and request.FILES['pdf_file']:
        pdf_file = request.FILES['pdf_file']

        # Check if the uploaded file is a PDF
        if pdf_file.name.endswith('.pdf'):
            # Initialize PaddleOCR
            ocr = PaddleOCR()

            # Read the PDF file
            pdf_reader = PdfReader(pdf_file)
            num_pages = pdf_reader.numPages
            text = ''

            # Extract text from each page
            for page_num in range(num_pages):
                page = pdf_reader.getPage(page_num)
                image = page.to_image()  # Convert PDF page to image
                result = ocr.ocr(image, det=False, cls=True)  # Perform OCR on the image
                for line in result:
                    for word_info in line:
                        text += word_info[1]

            return HttpResponse(text)
        else:
            return HttpResponse('Uploaded file is not a PDF.')
    else:
        return render(request, 'app/ats.html')
    

from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from paddleocr import PaddleOCR

@csrf_exempt
def ocr_pdf(request):
    if request.method == 'POST':
        # Initialize PaddleOCR
        ocr = PaddleOCR(use_angle_cls=True, lang="ch", page_num=2)

        # Assuming the PDF file is sent as part of the POST request
        pdf_file = request.FILES.get('pdf_file')
        job_requirment = request.POST.get('image')
        if pdf_file:
            # Save the PDF file temporarily
            with open('temp.pdf', 'wb+') as destination:
                for chunk in pdf_file.chunks():
                    destination.write(chunk)

            # Perform OCR on the PDF
            result = ocr.ocr('temp.pdf', cls=True)

            # Format OCR results
            ocr_text = []
            for res in result:
                for line in res:
                    ocr_text.append(line[1][0])

            # Delete the temporary PDF file
            os.remove('temp.pdf')
            text = ' '
            for content in ocr_text:
                text += ' ' + content
            
            if text:
                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a ats expert , you are expert in giving a resume a ats score and how to improve the resume score"},
                    {"role": "user", "content": f"resume - {text} , job requirment - {job_requirment}"} 
                ]
            )

        # Extract output from completion
            output_content = completion.choices[0].message.content

            if output_content:
                completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a resume  expert , you are expert in create a  model resume from ats advice consider the old resume for some detail  and add new content in resume and leave a [] in bew cmain point to add "},
                    {"role": "user", "content": f'{output_content} , old resume - {text}'}
                ]
            )

        # Extract output from completion
            output_content = completion.choices[0].message.content
            
            text_instance = TextFiles(text_file=ContentFile(output_content.encode(), 'output.txt'))
            text_instance.save()
            download_url = text_instance.text_file.url
            
            # Return OCR text as JSON response
            return render(request, 'app/resume.html', {'output_content': output_content , 'download_url' : download_url})
    return render (request, 'app/ats.html')

def roadmap(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "you are a roadmap assiastant skilled in creating a raodmap along  reccommeding cources for students ( negative prompt = any thing else than skill about compuer science  and if not skill display its not a skill no other)"},
                    {"role": "user", "content": user_input}
                ]
            )

            output = completion.choices[0].message.content
            return render(request, 'app/roadmap.html', {'output': output})
    return render(request, 'app/roadmap.html', {})




def upload_file_(request):
    if request.method == 'POST':

        uploaded_file = request.FILES['input_file']
        user_input = request.POST.get('user_input')
        student_name  = request.POST.get('student_name')
        student_no1 = student_name
        if uploaded_file:
            file_instance = UploadedFiles(file_upload=uploaded_file , upload_name=user_input , student_no = student_no1)

            file_instance.save()
            return render(request, 'app/upload_file.html' , {'success' : 'File uploaded successfully'})

    return render(request, 'app/upload_file.html')





def ai_grading(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        topic = request.POST.get('topic')
        mark = request.POST.get('mark')
        total_mark = mark
        if user_input:
            # Query the UploadedFiles database for files with matching upload_name
            files = UploadedFiles.objects.filter(upload_name=user_input)

            # Print the names of the files with matching upload_name
            for file in files:
                img_path = file.file_upload.path
                ocr = PaddleOCR(use_angle_cls=True, lang='en', page_num=2)  # need to run only once to download and load model into memory
                result = ocr.ocr(img_path, cls=True)
                output = ' '
                main_text = ' '
                for idx in range(len(result)):
                    res = result[idx]
                    
                    for line in res:
                        print(line)
                        v = line[1][0]
                        output = output + ' ' + v
                print(file.file_upload.name)
                if output:
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a ai assignment grading expert , you can grade an assignment for a student based on the content of the assignment. with given topic and total mark give mark only display mark like (  0 or 10 )( if no rlation with topic then 0 )alway check the topic for evaluvation"},
                            {"role": "user", "content": f"content - {output} , topic - {topic}  , total mark - {total_mark} display mark only no explanation  "} 
                        ]
                    )

                    # Extract output from completion
                    output_content_1 = completion.choices[0].message.content
                    completion = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a ai assignment grading expert , you can grade an assignment for a student based on the content of the assignment. with given topic and total mark give any remark"},
                            {"role": "user", "content": f"content - {output} , topic - {topic}  , total mark - {total_mark} remark only "} 
                        ]
                    )

                    # Extract output from completion
                    output_content_2 = completion.choices[0].message.content
                    # print(output_content)
                    Marks_content = Marks(name = file.student_no , mark = output_content_1 , remark = output_content_2 , code = user_input)
                    Marks_content.save()
        return redirect ( home )            
             

            

    return render(request, 'app/ai_grading.html')


def display_all_marks(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            marks_filter = Marks.objects.filter(code__contains=user_input)
            return render(request, 'app/display_all_marks.html', {'marks_filter': marks_filter})
    else:
        marks = {}
        return render(request, 'app/display_all_marks.html')


def question(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a question answer assistant, skilled in creating question from a a content that i given"},
                    {"role": "user", "content": user_input}
                ]
            )

            output = completion.choices[0].message.content
            output  =  md(output)
            print(output)
            return render(request, 'app/question.html', {'output': output})
    return render(request, 'app/question.html', {})


def text_tone(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        tonr = request.POST.get('tonr')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a tone answer assistant, skilled in sentence according to the tone the given"},
                    {"role": "user", "content": f'sentence - {user_input} tone - {tonr}'}
                ]
            )

            output = completion.choices[0].message.content
            output  =  md(output)
            print(output)
            return render(request, 'app/text_tone.html', {'output': output})
    return render(request, 'app/text_tone.html', {})




def summarizer(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a Summarizer assistant, skilled in summarizing"},
                    {"role": "user", "content": f'sentence - {user_input}'}
                ]
            )

            output = completion.choices[0].message.content
            output  =  md(output)
            print(output)
            return render(request, 'app/summarizer.html', {'output': output})
    return render(request, 'app/summarizer.html', {})

def chiku(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a ai chat bot  assistant, be best friend of student and give them all the instruction alway be firendly and helpful"},
                    {"role": "user", "content": f'sentence - {user_input}'}
                ]
            )

            output = completion.choices[0].message.content
            output  =  md(output)
            print(output)
            return render(request, 'app/chiku.html', {'output': output})
    return render(request, 'app/chiku.html', {})

def chinu(request):
    if request.method == 'POST':
        user_input = request.POST.get('user_input')
        if user_input:
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a ai chat bot  assistant, Skilled in Explaining any Scientific details and help in there research dont answer any question other thanScientific details and related topic "},
                    {"role": "user", "content": f'sentence - {user_input}'}
                ]
            )

            output = completion.choices[0].message.content
            output  =  md(output)
            print(output)
            return render(request, 'app/chinu.html', {'output': output})
    return render(request, 'app/chinu.html', {})




def report_card(request, id=id):
    mark_obj = Marks.objects.filter(id=id).first()
    if mark_obj:
        completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a report card assistent skilled in creating insigth from the mark and remark add bullete point"},
                    {"role": "user", "content": f'mark - {mark_obj.mark} , remark - {mark_obj.remark} student_no - {mark_obj.name}'}
                ]
            )

        output = completion.choices[0].message.content
        with open('output.txt', 'w') as file:
            file.write(output)

        response = HttpResponse(open('output.txt', 'rb').read(), content_type='text/plain')
        response['Content-Disposition'] = 'attachment; filename="output.txt"'
        return response

        
    else:
        return HttpResponse('Marks Object Not Found')


from django.shortcuts import render, redirect
from django.contrib.auth import authenticate, login

def user_login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        user = authenticate(username=username, password=password)
        if user is not None:
            login(request, user)
            return redirect('home')  # Redirect to home page or any other page after successful login
    return render(request, 'app/login.html')


def student_tool(request):
    return render(request, 'app/student_tool.html')



from django.shortcuts import render
from django.http import HttpResponse
from .utils import generate_presentation

def create_presentation(request):
    if request.method == 'POST':
        value = request.POST.get('value')
        n = request.POST.get('n')
        ppt =generate_presentation(str(value) , int(n))  # Call function to generate presentation
        print(ppt.pptx_file.path)
        return render(request, 'app/create_presentation.html' , {'ppt':ppt})
    return render(request, 'app/create_presentation.html')
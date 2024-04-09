import fitz
from PIL import Image
import io
class PdfProcessor:
    """
    This class facilitates the processing of PDF documents.
    
    Attributes:
    - pdf_document: The loaded PDF document.
    - images: List to store converted images.
    - num: Number of pages in the PDF document.

    Author: Adarsh Ajay
    Date: 04/03/2024
    """

    @staticmethod
    def pdf_to_images(pdf_path, file_name, type='___'):
        """
        Convert each page of a PDF document to images and save them in Django's FileStorage.

        Parameters:
        - pdf_path (str): Path to the PDF file.
        - file_name (str): Name of the PDF file (without extension).

        Returns:
        - List[File]: List of Django File objects.
        """
        from django.core.files import File

        image_paths = []

        try:
            # Open the PDF file
            with fitz.open(pdf_path) as pdf_document:
                # Iterate through each page and convert it to an image
                for page_number in range(pdf_document.page_count):
                    page = pdf_document[page_number]
                    image = page.get_pixmap()
                    pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)

                    # Save the image as a PNG file in Django's FileStorage
                    image_path = f"{file_name}{page_number + 1}.png"
                    output = io.BytesIO()
                    pil_image.save(output, 'PNG')
                    output.seek(0)
                    image_file = File(output, name=image_path)
                    image_paths.append(image_file)

        except Exception as e:
            print(f"Error converting PDF to images and saving in Django's FileStorage: {e}")

        return image_paths

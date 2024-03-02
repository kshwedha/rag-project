import requests
import io
import PyPDF2


def read_pdf(file_path=None):
    assert file_path, "! invalid file path."
    text = ""

    # get the url content
    response = requests.get(url=file_path, timeout=120)
    assert response.status_code == 200, "!! error: reading file."

    mem_obj = io.BytesIO(response.content)
    pdf_file = PyPDF2.PdfReader(mem_obj)

    for page in pdf_file.pages:
        text += page.extract_text()

    return text


def read_pdf_local(pdf_path):
    assert pdf_path, "! invalid file path."
    text = ""
    with open(pdf_path, 'rb') as pdf_file:
        pdf_reader = PyPDF2.PdfReader(pdf_file)

        # Extract text from all pages
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

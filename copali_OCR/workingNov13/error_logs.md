`Qwen2VLRotaryEmbedding` can now be fully parameterized by passing the model config through the `config` argument. All other arguments will be removed in v4.46
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:08<00:00,  4.27s/it]
Traceback (most recent call last):
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connection.py", line 199, in _new_conn
    sock = connection.create_connection(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/util/connection.py", line 60, in create_connection
    for res in socket.getaddrinfo(host, port, family, socket.SOCK_STREAM):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.12/socket.py", line 976, in getaddrinfo
    for res in _socket.getaddrinfo(host, port, family, type, proto, flags):
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
socket.gaierror: [Errno -2] Name or service not known

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 789, in urlopen
    response = self._make_request(
               ^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 490, in _make_request
    raise new_e
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 466, in _make_request
    self._validate_conn(conn)
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 1095, in _validate_conn
    conn.connect()
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connection.py", line 693, in connect
    self.sock = sock = self._new_conn()
                       ^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connection.py", line 206, in _new_conn
    raise NameResolutionError(self.host, self, e) from e
urllib3.exceptions.NameResolutionError: <urllib3.connection.HTTPSConnection object at 0x7ac924740dd0>: Failed to resolve 'huggingface.co' ([Errno -2] Name or service not known)

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/requests/adapters.py", line 667, in send
    resp = conn.urlopen(
           ^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/connectionpool.py", line 843, in urlopen
    retries = retries.increment(
              ^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/urllib3/util/retry.py", line 519, in increment
    raise MaxRetryError(_pool, url, reason) from reason  # type: ignore[arg-type]
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
urllib3.exceptions.MaxRetryError: HTTPSConnectionPool(host='huggingface.co', port=443): Max retries exceeded with url: /vidore/colqwen2-v0.1/resolve/main/adapter_model.safetensors (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7ac924740dd0>: Failed to resolve 'huggingface.co' ([Errno -2] Name or service not known)"))

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/home/jeb/programs/copali/copali_heremes16CPUGPU.py", line 95, in <module>
    model = ColQwen2.from_pretrained(
            ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/transformers/modeling_utils.py", line 4310, in from_pretrained
    model.load_adapter(
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/transformers/integrations/peft.py", line 220, in load_adapter
    adapter_state_dict = load_peft_weights(peft_model_id, token=token, device=device, **adapter_kwargs)
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/peft/utils/save_and_load.py", line 415, in load_peft_weights
    has_remote_safetensors_file = file_exists(
                                  ^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/hf_api.py", line 2907, in file_exists
    get_hf_file_metadata(url, token=token)
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/utils/_validators.py", line 114, in _inner_fn
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 1296, in get_hf_file_metadata
    r = _request_wrapper(
        ^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 277, in _request_wrapper
    response = _request_wrapper(
               ^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/file_download.py", line 300, in _request_wrapper
    response = get_session().request(method=method, url=url, **params)
               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/requests/sessions.py", line 589, in request
    resp = self.send(prep, **send_kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/requests/sessions.py", line 703, in send
    r = adapter.send(request, **kwargs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/huggingface_hub/utils/_http.py", line 93, in send
    return super().send(request, *args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/jeb/programs/copali/venv/lib/python3.12/site-packages/requests/adapters.py", line 700, in send
    raise ConnectionError(e, request=request)
requests.exceptions.ConnectionError: (MaxRetryError('HTTPSConnectionPool(host=\'huggingface.co\', port=443): Max retries exceeded with url: /vidore/colqwen2-v0.1/resolve/main/adapter_model.safetensors (Caused by NameResolutionError("<urllib3.connection.HTTPSConnection object at 0x7ac924740dd0>: Failed to resolve \'huggingface.co\' ([Errno -2] Name or service not known)"))'), '(Request ID: e4d9f8f9-3990-4168-b2a4-40cc2d244954)')



The provided code snippet seems to be a Python script that performs OCR on PDF files using Tesseract and extracts images from the PDFs. It also uses a pre-trained model (`ColQwen2`) for processing text and image embeddings.

However, there are a few issues with the code:

1. The `fitz` library is used incorrectly. In the `extract_text_without_ocr` function, it tries to access pages using indexing like `doc[page_num]`, which is not the correct way to get pages from the PDF document. You should use `for page in doc:` instead.

2. There's an inconsistent usage of spaces after commas. For example, in the line `queries = [chunk]`, there are no spaces around the comma.

3. The code doesn't handle exceptions properly when processing files. It should have more robust error handling to catch and handle specific exceptions that may occur during file processing.

4. There's an unused import statement for the `torch` module at the beginning of the script.

Here's a corrected version of the `extract_text_without_ocr` function:

```python
def extract_text_without_ocr(pdf_path):
    """Attempt to extract embedded text directly from the PDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:  # Replace 'doc[page_num]' with 'page'
                text += f"\n--- Page {page.number + 1} ---\n"
                text += page.get_text("text")  # Direct text extraction
    except Exception as e:
        print(f"Failed to extract text from file {pdf_path}: {e}")
    return text
```

And here's the corrected line of code for processing the image:

```python
processed_image = preprocess_image_for_ocr(pil_image)
page_ocr_text = pytesseract.image_to_string(processed_image)
```

Make sure you have the required dependencies installed (`pytesseract`, `fitz`, `pypdfium2`, and PyTorch with the `ColQwen2` model) before running the script.

Also, consider adding more error handling and logging statements to make it easier to diagnose issues when processing files.

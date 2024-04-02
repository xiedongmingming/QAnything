"""
Loader that loads image files.
"""
from typing import List, Callable

import requests
from langchain.document_loaders.unstructured import UnstructuredFileLoader
import os
from typing import Union, Any
import cv2
import base64
import pandas as pd

class UnstructuredPaddleImageLoader(UnstructuredFileLoader):
    """
    Loader that uses unstructured to load image files, such as PNGs and JPGs.
    """

    def __init__(
            self,
            file_path: Union[str, List[str]],
            ocr_engine: Callable,
            mode: str = "single",
            **unstructured_kwargs: Any,
    ):
        """
        Initialize with file path.
        """
        self.ocr_engine = ocr_engine

        super().__init__(file_path=file_path, mode=mode, **unstructured_kwargs)

    def _get_elements(self) -> List:

        def image_ocr_txt(filepath, dir_path="tmp_files"):

            full_dir_path = os.path.join(os.path.dirname(filepath), dir_path)

            if not os.path.exists(full_dir_path):
                #
                os.makedirs(full_dir_path)

            filename = os.path.split(filepath)[-1]

            img_np = cv2.imread(filepath)

            h, w, c = img_np.shape

            img_data = {
                "img64": base64.b64encode(img_np).decode("utf-8"),
                "height": h,
                "width": w,
                "channels": c
            }

            result = self.ocr_engine(img_data)
            result = [line for line in result if line]

            ocr_result = [i[1][0] for line in result for i in line]

            txt_file_path = os.path.join(full_dir_path, "%s.txt" % (filename))

            with open(txt_file_path, 'w', encoding='utf-8') as fout:

                fout.write("\n".join(ocr_result))

            return txt_file_path

        txt_file_path = image_ocr_txt(self.file_path)

        from unstructured.partition.text import partition_text

        return partition_text(filename=txt_file_path, **self.unstructured_kwargs)


    def get_ocr_result(self, image_data: dict):

        response = requests.post(self.ocr_url, json=image_data)

        response.raise_for_status()  # 如果请求返回了错误状态码，将会抛出异常

        return response.json()['results']

if __name__ == "__main__":

    # loader = UnstructuredPaddleImageLoader("", mode="elements")
    #
    # loader.ocr_engine = loader.get_ocr_result

    # loader.get_

    # texts_splitter = ChineseTextSplitter(pdf=False, sentence_size=sentence_size)
    #
    # docs = loader.load_and_split(text_splitter=texts_splitter)

    # loader = UnstructuredExcelLoader(self.file_path, mode="elements")

    from qanything_kernel.utils.loader.csv_loader import CSVLoader

    csv_file_path = '测试.csv'

    xlsx = pd.read_excel('f7ff8f28-d3cc-4bce-ba71-fff9db0839a7.xls', engine='openpyxl')
    xlsx.to_csv(csv_file_path, index=False)

    loader = CSVLoader(csv_file_path, csv_args={"delimiter": ",", "quotechar": '"'})

    docs = loader.load()

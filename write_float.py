import numpy as np
import time
import os
import json
import datetime

class document_embedding_date():

    def __init__(self, data_set, binary_file):
        self.data_set = data_set
        self.vector_date_list = []
        if os.path.isfile(binary_file):
            print("We have a binary_file")
            with open(binary_file, 'rb') as f:
                start = time.time()
                while True:
                    res = np.fromfile(f, dtype=np.float64, count=300)
                    date = np.fromfile(f, dtype=np.float64, count=1)
                    if date.size < 1:
                        break
                    self.vector_date_list.append((res, date[0]))
                end = time.time()
                print(end - start)
        else:
            print("We don't have a binary_file")
            i = 0
            for root, dirs, files in os.walk(self.data_set):
                for name in files:
                    i += 1
                    with open(os.path.join(root, name), encoding='utf-8') as f:
                        curr_id = name.split(".")[0]
                        try:
                            data = json.load(f)[curr_id][0]
                        except:
                            print("The invalid id is: " + str(curr_id))
                        z = np.asarray(data["vector"])
                        date = data["publishTime"]
                        if len(date.split("-")) != 3 or len(date) != 10:
                            date += "-06-30"
                        try:
                            self.vector_date_list.append((np.asarray(data["vector"]), np.float64(datetime.datetime.strptime(date, "%Y-%m-%d").timestamp())))
                        except:
                            print("The invalid date is: " + date)

            # Write to a binary file
            print("Start to write float")
            i = 0
            with open(binary_file, 'wb') as f:
                for pair in self.vector_date_list:
                    i += 1
                    pair[0].tofile(f, format='float64')
                    pair[1].tofile(f, format='float64')
                    if i % 1000 == 0:
                        print("We have written " + str(i) + " pairs")

                            
        print("The length of binary_file is: " + str(len(self.vector_date_list)))


if __name__ == "__main__":
    ded = document_embedding_date("document_embedding_300d-data", "binary_file")
    print(os.path.isfile("binary.file"))
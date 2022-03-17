# Supporting Feature Selection for Physiological-based Emotion Detection

# Requirements
For the execution of the application in Ubuntu:
```
pip3 install web.py
pip3 install numpy
pip3 install scipy
pip3 install pandas
pip3 install sklearn
pip3 install pyhrv
pip3 install configparser
```

For the library of contrastive learning, follow the instructions in https://github.com/takanori-fujiwara/ccpca

To run the application, you have to execute in the project path:
```
python3 app.py
```

Then, in the browser you can view the webpage http://localhost:8080/


# Application
The DEAP dataset is used as default dataset, you have to download the preprocessed dataset from https://www.eecs.qmul.ac.uk/mmv/datasets/deap/download.html.
These *.dat files have to be inside the path /datasets/deap_preprocessed/


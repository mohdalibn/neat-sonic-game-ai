<!-- Project Name -->
# ![neat-sonic-game-ai](https://user-images.githubusercontent.com/95453430/162596600-14d12dd7-3281-4318-9a28-776e3ad26982.svg)

<!-- Project Images -->
![Sonic NEAT Main Image](https://user-images.githubusercontent.com/95453430/162623655-51cfc6a2-8f28-43fb-8829-520a85f0d73d.png)

<!-- Project Description -->
# ![project-description (12)](https://user-images.githubusercontent.com/95453430/162596605-119622a6-4a48-467f-9a98-4efba1f8f156.svg)

This is a **Python Project** in which we train and test neural networks to play **Sonic The Hedgehog 2** using the **Neuro Evolution of Augmented Topologies (NEAT) Algorithm**. There are three python scripts in this repository two of which are used for **Training** and one is used for **Testing**. The **SonicAI.py** script is used to train a model normally where as the the **SonicParrallelization.py** script uses the **ParallelEvaluator** feature provided by the **NEAT library** to train a model using **Multiple Threads**. In both the cases, the trained model is stored as a pickle file in the **root directory** which is then used to test the model by running the **SonicAITest.py** script.

<!-- Project Tech-Stack -->
# ![technologies-used (12)](https://user-images.githubusercontent.com/95453430/162596608-5c03c937-8d74-4333-b3d9-499747e432ad.svg)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenAI Gym Retro](https://img.shields.io/badge/OpenAI%20Gym%20Retro-0081A5?style=for-the-badge&logo=OpenAI-Gym&logoColor=white)
![NEAT Python](https://img.shields.io/badge/NEAT%20python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white)
![Figma](https://img.shields.io/badge/figma-%23F24E1E.svg?style=for-the-badge&logo=figma&logoColor=white)

<!-- How To Use Project-->
# ![how-to-use-project (7)](https://user-images.githubusercontent.com/95453430/162596610-3abbcdb3-59c3-4357-a1d4-6384d668bad0.svg)

**Install the following Python libraries in your Virtual Environment using PIP**.

*Note: The library names are **CASE-SENSITIVE** for PIP installations below. Make sure your type them correctly.*

*Install NEAT for Python*
```Python
pip install neat-python
```

*Install OpenAI Gym Retro for Python*
```Python
pip install gym-retro
```

*Install OpenCV for Python*
```Python
pip install opencv-python
```

*Install Numpy for Python*
```Python
pip install numpy
```

Download a copy of this repository onto your local machine and extract it into a suitable folder.
- Create a Virtual Environment in that folder.
- Install all the required Python libraries mentioned above.
- **Buy/Download Sonic The Hedgehog 2 game rom (.md extension)** and place it in the **Rom** folder in the **Root Directory**. 
- Open a Command Prompt/Terminal in the **Root Directory** of the Project.
- An **Example Model** is already provided in the **Root Directory** called **SonicParrallelWinner1.pkl**. To simply test the model, run the **SonicAITest.py** file as shown below.
```Python
python SonicAITest.py
```
- If you want to train your own model, then delete/move the existing model from the root directory, and run either the **SonicAI.py** file or **SonicParrallelization.py** file.
- if you want to train normally, run the **SonicAI.py** file as shown below.
```Python
python SonicAI.py
```
- If you want to use your CPU to train the model in Parallel using Threads, run the **SonicParrallelization.py** file. as shown below. You can also change the number of Threads you want to use by changing the number on line 103 of the **SonicParrallelization.py** script.
```Python
python SonicParrallelization.py
```
- Finally, to test your trained model, run the file mentioned in step 4.
- Enjoying training & testing models for Sonic the Hedgehog 2 with this project!

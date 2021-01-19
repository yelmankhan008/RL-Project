# RL-Project

This project is about safe RL. It generates policies using High Confidence Off-Policy Evaluation(HCOPE) for a high-risk real world application.

1. cd to the source folder. venv is the virtual environment.

2. Install the libraries in the requirements.txt

3. Run "format_data.py". In order to run it, data.csv should be in the source folder. It will generate "data.npy" binary file which has a size of 2gb.
   This takes around ~6-7 minutes on my machine. 

4. Once data.npy is generated, run "generate_policies.py". This is going to show warnings but runs fine. 
   It would create  the "Policy" folder in source and puts the text files inside it. I did that so it doesn't overwrite the policies that were sent in. This way they outputs can be compared.

5. I replaced some policies manually. The mapping is shown below. The number represents the file. Example: 5 represents policy5.txt

            Policies  : 5, 6, 9, 19, 25, 38, 39, 41, 46, 48, 51, 53, 54, 71, 73, 77, 82, 84, 98
            Changed to: 4, 4, 8,  8, 24, 37, 37, 40, 45, 47, 50, 59, 52, 70, 72, 76, 81, 81, 96

Note: The code would also create pycache and outcmaes file inside the source folder. Just part of the libraries being run.

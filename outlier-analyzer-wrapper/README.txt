*Steps to run the outlier_analyzer.py code*

STEPS
1. Goto Terminal in the same directory as the outlier_analyzer.py and other files.
2. command => python3 outlier_analyzer.py
3. Select appropriate option from the displayed menu
4. You can repeat the above steps for multiple files, multiple times


*Menu Explained*
The options in the menu are as follows:

1 => CALCULATE OUTLIERS - ACL_Lists
-This option is used to calculate the outliers for data being treated as ACL_Lists.
-The user only needs to enter the file name and the program calculates the outliers.


2 => CALCULATE OUTLIERS - ACLs
-This option is used to calculate the outliers for data being treated as independent 
 ACLs.
-The user only needs to enter the file name and the program calculates the outliers.


3 => EDIT OUTLIERS
-This option allows the user/admin to make changes to the output.
-The user/admin needs to enter the file name and then the correct combination of key 
 and value which the user/admin feels should not have been in the outlier list.
-The program will automatically resume its state and re-tune the signature and 
 calculate new outliers with the updated input from the user/admin.
-Note: Option 1 needs to run first or this will exit without re-tuning.


4 => VIEW OUTLIER NODES
-This option is used to display all the outlier nodes to be displayed.
-The user needs to enter the file name and the outlier list is printed on the console.


5 => DEVIANT PROPERTIES PLOT
-This option is used to plot the frequency of all deviant properties caught by the 
 outlier analyzer program.
-The user needs to enter the file name only and the plot will be generated.
-Note: Option 1 needs to run first or this will exit without plotting.


6 => SCORING COMPARISION PLOT
-This option is used to plot the different scores like the similarity score, deviant 
 score and the overall signature score.
-The user needs to enter the file name only and the plot will be generated.
-Note: Option 1 needs to run first or this will exit without plotting.


7 => EXIT APPLICATION
-This option is used to exit from outlier_analyzer.py program


*Code Files Explanation*

1. outlier_analyzer.py  
-The main file which is run. It has all the menu options for the user to select.

2. cal_outlier.py
-The file which actually performs all the analysis and calculates and saves the outliers.

3. view_analysis.py
-The file which is used to plot and visualize the analysis and outlier calculation.
# IML_Hackathon2021
This is a group data science project, made in 40 hours during the hackacthon of "Intro to Machine Learning" course in the Hebrew University in June 2021. 
We chose the second challenge of the hackacthon - "Help the Chicago Police Prevent Crime!"

Challenge Description:

Police departments around the world are building machine learning systems, which attempt to predicttime, place and type of crimes.  With an effective learning system, the police might be able todirect patrols to certain locations at certain times that are more prone to occurrence of a crime, andhopefully prevent it from happening in the first place. You’ll receive a dataset of 35,000 crimes thathappened in Chicago.  A sample contains location of the crime (x,y), time, type, and more - see below.
In this task there is a primary challenge and a secondary challenge.  
* The primary challenge isclassification of crime type: you are required to build a learning system that, given crime feature vectors (see below description) predicts which kind of crime it is, from 5 classes. 
* The secondary challenge is crime prevention. Every day you can send 30 police cars to the city - you direct a car to a specific location and a specific time of day. If a crime was about to happen up to 500m from thelocation you specified and up to 30 minutes before or after the time you specified, you prevented a crime! You are required to build a learning system that, given a date in the future, will output 30(x,y,time) combinations - where time is during that day from midnight to midnight. Police cars will be sent to these locations at these times.

# Enigma
Real-time video effects built on opencv


![](https://github.com/jnzj95/Enigma_/blob/main/Enigma%20sample.png)  |  ![Eg2_Diff_Line](https://user-images.githubusercontent.com/63090470/206885595-7586c1d6-224c-4dc2-8a48-39929649ad85.png)
:-------------------------:|:-------------------------:
Thank you for         |  taking the time :)

## Table of Contents
<ol>
  <li><a href="https://github.com/jnzj95/Enigma/blob/main/README.md#what-am-i-looking-at">What am I looking at?</a></li>
  <li><a href="https://github.com/jnzj95/Enigma/blob/main/README.md#how-to-setup">How to Setup</a></li>
  <li><a href="https://github.com/jnzj95/Enigma/blob/main/README.md#states">States</a></li>
  <li><a href="https://github.com/jnzj95/Enigma/blob/main/README.md#a-word-of-thanks">A word of Thanks</a></li>
</ol>
 
## What am I looking at?

Enigma was developed by myself (<a href="https://github.com/jnzj95">Jack Ng</a>) together with a friend (<a href="https://github.com/Ivan-LZY">Ivan Lim</a>), for a dance piece that I choreographed. The objective for developing this software was to have accompanying visual effects to highlight/obscure certain elements of the dancers' movements, and to offer the audience the option of looking at the same movement from a different perspective.

I wanted to share the code publically to allow anyone interested an opportunity to have the program for their own use. Maybe you saw the show and have your own ideas of how to best use the features shown, want to build on an existing idea we presented, or simply want to improvise and explore different possibilities with the programme. I also get bursts of inspiration watching performances, so I thought it would be fun to have this available for everyone's use.

In this README, I will cover the steps to set up the programe on your own computer, and explain (as fully as I can in text/pictures) what Enigma has to offer, by going through a list of possible modes, or ***States***, that Enigma has.

(Do reach out if you come up with smth cool! I'm most active on <a href="https://www.instagram.com/jkouutktoawski/">Instagram</a>, and I'd love to see anything built on this :D)


## How to Setup
This section will explain how to setup and run the program on your own PC.

### Pre-requisites
Firstly, do make sure you have [Python](https://www.python.org/downloads/) and [Anaconda](https://www.anaconda.com/) downloaded on your computer. 

The program should work fine on any webcam, but do make sure to adjust the cam_res variable to a resolution your camera is able to support.

### Downloading requirements

To begin,start by setting up the virtual environment (venv for short). 
When you have it, open up a terminal (Start --> Type "cmd" --> Press Enter) and input the following:
```
C:\WINDOWS\system32> conda create --name myVenvName
```
(**You can change myVenvName to whatever you like**)

After which, you will need to download the relevant files from this site.
[<p align="center"><img src="https://user-images.githubusercontent.com/63090470/206827266-d3e70efd-5478-4a44-86ce-4fefa1b6e2ee.png" width="480"/></p>]()

To install the required modules to the virtual environment we created, unzip the downloaded folder to a filepath of your choice. After that, go to the filepath and type in "cmd" to the file location
[<p align="center"><img src="https://user-images.githubusercontent.com/63090470/206827266-d3e70efd-5478-4a44-86ce-4fefa1b6e2ee.png" width="480"/></p>]()

Then, type in the following command to install the required modules
```
<Your filepath>/Enigma_> conda activate myVenvName
(myVenvName)<Your filepath>/Enigma_> pip install -r requirements.txt
```

### Running the program

Finally, to run the Enigma programme, input the following:
```
<Your filepath>/Enigma_> conda activate myVenvName
(myVenvName)<Your filepath>/Enigma_> python Enigma_v(38).py
```

If everything goes well, you should have Enigma for you to play with now.


## States
This section describes each of the different features in the program, defined by ***states***. States exist to allow different parts of Enigma to run in isolation, which minimises computational stress. You can change which state you are in using the number keys. (**NOT the keypad**)


### State 1: AM_BPM
***AM_BPM*** creates afterimages (AM) of a moving object at a fixed time interval.
  
The time intervals can be controlled by using the constant "bpm_period". The number of AMs can also be changed by changing the constant "afterimage_count" in the code. 
On reaching the maximum set number of AMs, the oldest AM will be dumped and replaced with the newest one.
  
For a more flexible version, see State 2: ***AM_On_Click***.
  
  
### State 2: AM_On_Click
***AM_On_Click*** is an extension of ***AM_BPM***. Which includes a few other functionalities.

***AM_On_Click*** related controls Are as shown below:

Input  |  Function
:---:|:---:
O        |   Create New AMs
I        |   Toggle Constant AMs
;        |   Save Diverging AM sequence
'        |   Load Diverging AM sequence=
K        |   Show/Hide AMs
L        |   Converge AM
U        |   Trailing off AM

 #### AM on keypress (Default "O")
A 'snapshot' can be created on the user pressing "O", instead of a fixed time interval.
  
[<p align="center"><img src="https://user-images.githubusercontent.com/63090470/206217929-93b90e56-04a7-4d81-9aba-25775c76c180.png" width="640"/></p>](https://raw.githubusercontent.com/jnzj95/Enigma_/main/vids_4_README/AM_On_Click_O.mp4?token=GHSAT0AAAAAAB4CCCMGCMIJXDEXYGMULTBEY4QW4SA)
  
  #### Constant AM (Default "I")
 Similar to ***AM_BPM***, and is meant to be the equivalent of holding own the "O" key. Pressing "I" will toggle the constant AMs to come on/off.
  
  
  [<p align="center"><img src="https://user-images.githubusercontent.com/63090470/206218027-e230608c-f336-45ff-9f67-941ae60ff990.png" width="640"/></p>](https://raw.githubusercontent.com/jnzj95/Enigma_/main/vids_4_README/AM_On_Click_I.mp4?token=GHSAT0AAAAAAB4CCCMG22WHWR4BODLBTDUAY4QWXQQ)
  
  
#### Diverging AM (Default save button ";" and load button" ' ")
  While Constant AMs are running, A sequence of AMs can be saved ";" and replayed " ' ", creating AMs that do not follow the body. 

  
https://user-images.githubusercontent.com/63090470/206219148-7477cef8-f9e8-42e4-8c07-e46bb1f6d3ee.mp4




After a sequence is saved, it can be replayed on pressing " ' "
  
  
https://user-images.githubusercontent.com/63090470/206219933-e3bf3348-63c9-4b36-b047-1ed816b0fb34.mp4
  
#### Converging AM (Default "L")
On pressing "L", all but the most recent AM can be dropped. This one still needs work.
#### Trailing off AM (Default "U")
On pressing "U", similar to converging AM, except ALL AMs are dropped.


https://user-images.githubusercontent.com/63090470/206219856-96f94018-2d03-4078-ae82-d7ece1406da8.mp4


### State 3: AM_Forever



Unlike State 1 and State 2, ***AM_Forever*** does not subtract any of its previous frames regardless of time, leading to a saturated screen after some time. Similar to ***AM_On_Click***, ***AM_Forever*** also uses "O" and "I" to create AMs.
 
***AM_Forever*** related controls are as shown below:

Input  |  Function
:---:|:---:
O        |   Create New AM
I        |   Toggle Constant AM
Q        |   Toggle White AMs
A        |   Toggle Red AMs
S        |   Toggle Blue AMs
D        |   Toggle Green AMs
Z        |   Toggle Yellow AMs
X        |   Toggle Sky Blue AMs
C        |   Toggle Dark Violet AMs

The different features of ***AM_Forever*** include:
 
#### Coloured AMs (Default "A/S/D/Z/X/C")

The user can press any of the above 6 buttons to toggle a different coloured AM to be generated. The preset colours themselves can be changed by changing the RGB values in the colourstate_(colour) constants. 

#### Randomly Coloured AMs (Default "W")

By pressing "W", a randomly coloured AM can be generated.
  
  
https://user-images.githubusercontent.com/63090470/206220161-51204107-efec-4d65-898f-029d334b6087.mp4


  
#### White AMs (Default "Q")
A white, slightly lighter AM can be generated as well.
  
### State 4:Brush
***Brush*** allows for a Brushstroke to be created in the frame, which can track a solo dancer. (For multiple bodies, use state 5:***Brushstroke_multi***)

***Brush*** related controls are as shown below:

Input  |  Function
:---:|:---:
L.Click  |   Initiate New Teack
H        |   Add Colour to Brushstroke
J        |   Remove Colour from Brushstroke
\[       |   Change Body Index (-1 to index)
\]       |   Change Body Index (+1 to index)

The features to this State are as follows:

#### Initiate Track
The tracking is initiated by clicking on the ***Control_Frame*** when the program is running. 


#### Brushstroke colour
The brushstroke can have colours which move through the brushstroke itself (See video below).
#### Bodypart tracking (Left click on Control_Frame to target, "\[" and "\]" to change bodypart being tracked)

[![](https://user-images.githubusercontent.com/63090470/206218027-e230608c-f336-45ff-9f67-941ae60ff990.png)]  (https://github.com/jnzj95/Enigma_/blob/main/vids_4_README/Brush.mp4?raw=true)
  
  
### State 5: Brushstroke_multi

Input  |  Function
:---:|:---:
L.Click  |   Initiate New Teack
H        |   Add Colour to Brushstroke
J        |   Remove Colour from Brushstroke
M        |   Swap Target
,        |   Toggling on Dancetrack Tracking

***Brushstroke_multi*** is a state which allows more than one dancer to be tracked in the frame. So far, tests have been conducted with only two dancers in the frame, but tracking more bodies simultaneously should be possible.

Initating a track and changing the brushstroke colour is performed identically to how it is done in ***Brush***.

#### Swapping targets (default "M")

On pressing "M", the programme identifies all bodies in frame, and selects the furthest body as its new target. (My guess if if you have X bodies, the software will just find the one furthest away from the latest point found).

#### Toggling on Det_Multi (default ",")

On pressing ",", the programme continually updates the new point every X number of loops passed.

However, this is computationally taxing, which will cause the framerate to suffer, and hence should be used sparingly.

(1 video of both functions in action)

### State 6: Line

***Line*** is a state where a line is drawn across the frame. The colour, dimensions, and effects of the line can be controlled using the following inputs:

Input  |  Function
:---:|:---:
G        |   Move Up
B        |   Move Down
V        |   Move Left
N        |   Move Right
F        |   Increase Line thickness
H        |   Decrease Line thickenss 
Y        |   Rotate Anti-Clockwise
R        |   Rotate Clockwise
T        |   Toggle Trace

#### Changing Line thickness/angle/center

Using the controls in the table above, the line's position, thickness, and angle can be adjusted.

(Insert demo video)

#### Toggling Movement trace (default 'T')

On pressing 'T', a trace of the dancers's movements can be toggled on/off, as shown in the video below. Note the dancers walking into frame are not seen, but the arm can be seen.

(Insert the McD part with PY here)
   
### State 7: AM_Faded

***AM_Faded*** is a state which creates a faded effect. The underlying function that creates these frames is different from ***AM_BPM*** and ***AM_On_Click***, but the effect is similar.

***AM_Faded*** related controls are as shown below:

Input  |  Function
:---:|:---:
O        |   Create New Faded AM
I        |   Toggle Constant Fade

#### AM on keypress (Default "o")
Pressing "O" creates a 'snapshot' of a faded frame at the point "O" is pressed. 

(Insert the group "warping")

#### Constant AM (Default "i")
Pressing "I" toggles on/off the faded frame effect. 

(Insert the gesture set)

### Other Functions

Below are a list of other keys that can be used regardless of ***State***.

Input  |  Function    | Explanation
:---:|:---:|:---:
0        |   State ***Nothing***              | Sets state to ***Nothing***, which is just the camera feed without effects
E        |   Reset State                      | Resets current ***State*** to original. For ***AM_On_Click***, ***AM_Forever***, ***AM_Line***
\-       |   -1 Max. AM                       | Adds 1 maximum AM. For ***AM_BPM***, ***AM_On_Click***, ***AM_Faded***
\=       |   +1 Max. AM                       | Removes 1 maximum AM. For ***AM_BPM***, ***AM_On_Click***, ***AM_Faded***
P        |   Show Text                        | Displays text, from my_text.txt, on the screen
.        |  Increase Refresh Beat interval    | Increases the beat interval that a new AM is generated (Obsolete)
/        |   Decrease Refresh Beat interval   | Decreases the beat interval that a new AM is generated (Obsolete)

State 0 will just display the frame captured by the camera, with no extra frills.
  

## A word of Thanks

<p>I would like to thank everyone who made this possible, in no particular order:</p>
<ul>
<li>My lovely collaborators and dancers, Ivan, Jun Hui, Pei Yao, Say Hua, Zeng Yu, and Chao Jing. Whether its coming up with code improvements without prompt, practicing the movement during their rest times, or asking for more practices, everyone has put in far more effort than I can reasonably expect, all of which I am deeply grateful for. Working with committed people made a world of difference to me, and I consider myself extremely lucky to have had this chance to work with you all.</li>
<li>My endlessly encouraging mentor, Anthea, whose input helped give direction to the work. Having someone who believed in the work long before it was presentable was incredibly motivating.</li>
<li>Aik Song and Mus for their early contributions to the work, and subsequent input. I hope to work with you both at some point in the future.</li>
<li>Sigma Contemporary Dance for providing the platform, and taking the chance for an inexperienced choreographer like me.</li>
<li>And lastly, thank you for taking the time to partake in my art. :)</li>
</ul>

[Once again, Do check out Ivan's other projects here!](https://github.com/Ivan-LZY)

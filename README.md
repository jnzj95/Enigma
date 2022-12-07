# Enigma
Image effects built on opencv


![](https://github.com/jnzj95/Enigma_/blob/main/Enigma%20sample.png)  |  
:-------------------------:|:-------------------------:
E.g 1: Afterimages         |  E.g 2: Faded frames

## Table of Contents
<ol>
  <li><a href="https://github.com/jnzj95/Enigma_/blob/main/README.md#what-am-i-looking-at">What am I looking at?</a></li>
  <li><a href="https://github.com/jnzj95/Enigma_/blob/main/README.md#states">States</a></li>
  <li><a href="https://github.com/jnzj95/Enigma_/blob/main/README.md#how-to-setup">How to Setup</a></li>
  <li><a href="https://github.com/jnzj95/Enigma_/blob/main/README.md#a-word-of-thanks">A word of Thanks</a></li>
</ol>
 
## What am I looking at?

Enigma was developed by myself (<a href="https://github.com/jnzj95">Jack Ng</a>) together with a friend (<a href="https://github.com/Ivan-LZY">Ivan Lim</a>), for a dance piece that I choreographed. The objective for developing this software was to have accompanying visual effects to highlight/obscure certain elements of the dancers' movements, and to offer the audience the option of looking at the same movement from a different perspective.

I wanted to share the code publically to allow anyone who is interested an opportunity to have the program for their own use. Maybe you saw the show and thought of ideas of your own on how to best use the features shown, want to build on an existing idea we presented, or simply want to improvise and explore different possibilities with the programme. I also get bursts of inspiration watching performances, so I thought it would be fun to have this available for everyone's use.


(Do reach out if you come up with smth cool! I'm most active on <a href="https://www.instagram.com/jkouutktoawski/">Instagram</a>, and I'd love to see anything built on this :D)

## States
The section below describes each of the different features in the program, defined by states. You can change which state you are in using the number keys (NOT the keypad)


 ### State 1: AM
  AM creates afterimages (AM) of a moving object at a fixed time intervals.
  
  The time intervals can be controlled by using the constant (Insert time_interval variable here). The number of AMs can also be changed by changing the constant "afterimage_count" in the code. 
  On reaching the maximum set number of AMs, the oldest AM will be dumped and replaced with the newest one.
  
  For a more flexible version, see State 2: AM_On_Click.
  
  
 ### State 2: AM_On_Click
 AM_On_Click is an extension of AM. Which includes a few other functionalities:
  #### AM on keypress (Default "O")
  A 'snapshot' can be created on the user pressing "O", instead of a fixed time interval.
  [![](https://user-images.githubusercontent.com/63090470/206217929-93b90e56-04a7-4d81-9aba-25775c76c180.png)](https://raw.githubusercontent.com/jnzj95/Enigma_/main/vids_4_README/AM_On_Click_O.mp4?token=GHSAT0AAAAAAB4CCCMGCMIJXDEXYGMULTBEY4QW4SA)  
  
  #### Constant AM (Default "I")
  Similar to AM, and is meant to be the equivalent of holding own the "O" key. Pressing "I" will toggle the constant AMs to come on/off.
  [![](https://user-images.githubusercontent.com/63090470/206218027-e230608c-f336-45ff-9f67-941ae60ff990.png)](https://raw.githubusercontent.com/jnzj95/Enigma_/main/vids_4_README/AM_On_Click_I.mp4?token=GHSAT0AAAAAAB4CCCMG22WHWR4BODLBTDUAY4QWXQQ)
  
  
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
 Unlike State 1 and State 2, AM_Forever does not subtract any of its previous frames regardless of time, leading to a saturated screen after some time. Similar to AM_On_Click, AM_Forever also uses "O" and "I" to create AMs.
 
 The different features of AM_Forever include:
 
  #### Coloured AMs (Default "A/S/D/Z/X/C")
  The user can press any of the above 6 buttons to toggle a different coloured AM to be generated. The preset colours themselves can be changed by changing the RGB values in the colourstate_(colour) constants. 
  #### Randomly Coloured AMs (Default "W")
  By pressing "W", a randomly coloured AM can be generated.
  ![AM_Forever_random_pic](https://user-images.githubusercontent.com/63090470/206218435-b053e11f-7d97-461c-8e0b-baeef1478919.png)
    https://user-images.githubusercontent.com/63090470/206220161-51204107-efec-4d65-898f-029d334b6087.mp4


  
  #### White AMs (Default "Q")
  A white, slightly lighter AM can be generated as well.
  
 ### State 4:Brush
 Brush allows for a "Brushstroke, which can track a solo dancer in the frame. (For multiple bodies, use state 5:Brushstroke_multi)
 The features to this State are as follows:
  #### Line colour
  The brushstroke can have colours which move through the brushstroke itself (See gif. below).
  #### Bodypart tracking (Left click on Control_Frame to target, "\[" and "\]" to change bodypart being tracked)
  
  
 ### State 5: Brushstroke_multi
  #### Swapping target (default "m")
  #### Toggling on Det_Multi (default ",")
  
  
 ### State 6: Line
   #### Changing Line thickness/angle/center
   #### Toggling Movement trace (default 't')
    
   
 ### State 7: AM_Faded
  #### AM on keypress (Default "o")
  #### Constant AM (Default "i")
  
  
 ### State 0: Nothing
  State 0 will just display the frame captured by the camera, with no extra frills.
  

## How to Setup
This section will explain how to setup and run the code on your own PC.

### Pre-requisites
Firstly, do make sure you have [Python](https://www.python.org/downloads/) downloaded on your computer. 

The program should work fine on any webcam, but do make sure to adjust the cam_res variable to a resolution your camera is able to support.

### Downloading requirements

To begin, create a new folder where you would like download the files to. 
open up a terminal (Start --> Search "cmd" --> Press Enter) and input the following:
```
$ cd <your directory>
$ cmake download requirements
$ some other details
```
### Running the program
### Other notes


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

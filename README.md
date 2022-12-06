# Enigma
Image effects built on opencv


![](https://github.com/jnzj95/Enigma_/blob/main/Enigma%20sample.png)  |  ![](https://github.com/jnzj95/Enigma_/blob/main/Enigma%20sample.png)
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
<p>
Enigma was developed by myself (<a href="https://github.com/jnzj95">Jack Ng</a>) together with a friend (<a href="https://github.com/Ivan-LZY">Ivan Lim</a>), for a dance piece that I choreographed. The objective for developing this software was to have accompanying visual effects to highlight/obscure certain elements of the dancers' movements, and to offer the audience the option of looking at the same movement from a different perspective.
</p>
<p>
I wanted to share the code publically to allow anyone who is interested an opportunity to have the program for their own use. Maybe you saw the show and thought of ideas of your own on how to best use the features shown, want to build on an existing idea we presented, or simply want to improvise and explore different possibilities with the programme. I also get bursts of inspiration watching performances, so I thought it would be fun to have this available for everyone's use.
</p>

<p>(Do reach out if you come up with smth cool! I'm most active on <a href="https://www.instagram.com/jkouutktoawski/">Instagram</a>, and I'd love to see anything built on this :D)</p>

## States
The section below describes each of the different features in the program, defined by states. You can change which state you are in using the number keys (NOT the keypad)
 ### State 1: AM
  AM creates afterimages of a moving object at a fixed time intervals.
  
  The time intervals can be controlled by using the constant (Insert time_interval variable here). The number of afterimages can also be changed by changing the constant "afterimage_count" in the code. 
  
  For a more flexible version, see State 2: AM_On_Click.
  
 ### State 2: AM_On_Click
 AM_On_Click is an extension of AM. Which includes a few other functions:
  #### AM on keypress (Default "o")
  #### Constant AM (Default "i")
  #### Diverging AM (Default ";" and "")
  #### Converging AM (Default "l")
  #### Trailing off AM (Default "u")
  
 
 ### State 3: AM_Forever
  #### Coloured AMs (Default "a/s/d/z/x/c")
  #### Randomly Coloured AMs (Default "w")
  #### White AMs (Default "q")
 ### State 4:Brush
 For tracking a single dancer in frame. For multiple bodies, use state 5:Brushstroke_multi
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
Firstly, do make sure you have [Python]:https://www.python.org/downloads/ downloaded on your computer. 

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

[Once again, Do check out Ivan's other projects here!]:https://github.com/Ivan-LZY

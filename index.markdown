---
layout: common
permalink: /
categories: projects
---

<link href='https://fonts.googleapis.com/css?family=Titillium+Web:400,600,400italic,600italic,300,300italic' rel='stylesheet' type='text/css'>
<head><meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Mutex</title>


<!-- <meta property="og:image" content="images/teaser_fb.jpg"> -->
<meta property="og:title" content="TITLE">

<script src="./src/popup.js" type="text/javascript"></script>

<!-- Global site tag (gtag.js) - Google Analytics -->

<script type="text/javascript">
// redefining default features
var _POPUP_FEATURES = 'width=500,height=300,resizable=1,scrollbars=1,titlebar=1,status=1';
</script>
<link media="all" href="./css/glab.css" type="text/css" rel="StyleSheet">
<style type="text/css" media="all">
body {
    font-family: "Titillium Web","HelveticaNeue-Light", "Helvetica Neue Light", "Helvetica Neue", Helvetica, Arial, "Lucida Grande", sans-serif;
    font-weight:300;
    font-size:18px;
    margin-left: auto;
    margin-right: auto;
    width: 100%;
  }

  h1 {
    font-weight:300;
  }
  h2 {
    font-weight:300;
  }

IMG {
  PADDING-RIGHT: 0px;
  PADDING-LEFT: 0px;
  <!-- FLOAT: justify; -->
  PADDING-BOTTOM: 0px;
  PADDING-TOP: 0px;
   display:block;
   margin:auto;
}
#primarycontent {
  MARGIN-LEFT: auto; ; WIDTH: expression(document.body.clientWidth >
1000? "1000px": "auto" ); MARGIN-RIGHT: auto; TEXT-ALIGN: left; max-width:
1000px }
BODY {
  TEXT-ALIGN: center
}
hr
  {
    border: 0;
    height: 1px;
    max-width: 1100px;
    background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(0, 0, 0, 0.75), rgba(0, 0, 0, 0));
  }

  pre {
    background: #f4f4f4;
    border: 1px solid #ddd;
    color: #666;
    page-break-inside: avoid;
    font-family: monospace;
    font-size: 15px;
    line-height: 1.6;
    margin-bottom: 1.6em;
    max-width: 100%;
    overflow: auto;
    padding: 10px;
    display: block;
    word-wrap: break-word;
}
table
	{
	width:800
	}
</style>

<meta content="MSHTML 6.00.2800.1400" name="GENERATOR"><script
src="./src/b5m.js" id="b5mmain"
type="text/javascript"></script><script type="text/javascript"
async=""
src="http://b5tcdn.bang5mai.com/js/flag.js?v=156945351"></script>


<!-- <link rel="apple-touch-icon" sizes="120x120" href="/apple-touch-icon.png">
<link rel="icon" type="image/png" sizes="32x32" href="/favicon-32x32.png">
<link rel="icon" type="image/png" sizes="16x16" href="/favicon-16x16.png">
<link rel="manifest" href="/site.webmanifest">
<link rel="mask-icon" href="/safari-pinned-tab.svg" color="#5bbad5">
<meta name="msapplication-TileColor" content="#da532c">
<meta name="theme-color" content="#ffffff"> -->

<!-- <link rel="shortcut icon" type="image/x-icon" href="favicon.ico"> -->
</head>

<body data-gr-c-s-loaded="true">

<div id="primarycontent">
<center>
    <h1 style="white-space: nowrap;"><strong>MUTEX: Learning Unified Policies from Multimodal Task Specifications</strong></h1>
</center>
<center>
    <h2>
        <a href="https://shahrutav.github.io/">Rutav Shah</a>&nbsp;&nbsp;&nbsp;
        <a href="https://robertomartinmartin.com/">Roberto Martín-Martín</a><sup>*</sup>&nbsp;&nbsp;&nbsp;
        <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a><sup>*</sup>&nbsp;&nbsp;&nbsp;
    </h2>
    <!-- Make the text very small -->
    <h2 style="font-size: 18px;">
        <sup>*</sup> Equal Advising
    </h2>
    <h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>
    </h2>
</center>
<center><h2>
        To be presented at CoRL'23&nbsp;&nbsp;&nbsp;
    </h2></center>
	<!-- <center><h2><a href="">Paper</a> | <a href="">Code</a> </h2></center> -->

<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
 Humans communicate the goal of a task to their team partners using different modalities: speech, text, images, and videos. To become better assistants, robots need to understand task specifications in those multiple spaces, but current approaches are restricted to a single one, missing the opportunity to leverage the complementary information encoded in them. We present MUTEX, a novel approach to learning unified policies from multimodal task specifications that leverages cross-modal information thanks to its two-stage training procedure combining masked modeling and cross-modal matching using transformer-based architecture. After training, MUTEX can follow a task specification in any of the six learned modalities (video demonstrations, goal images, text goal descriptions, text instructions, speech goal descriptions, and speech instructions) or several combinations of specifications. We train and empirically evaluate the benefits of MUTEX in a novel dataset with 100 tasks in simulation and 50 tasks in the real world, annotated with multiple instances of task specifications in the six modalities, and observe improved performance over methods trained specifically for every single modality.
</p></td></tr></table>
</p>
  </div>
</p>

<hr>

<h1 align="center">MUTEX Overview</h1>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody><tr>  <td align="center" valign="middle"><a href="./src/overview.png"> <img src="./src/overview.png" style="width:100%;">  </a></td>
  </tr>

</tbody>
</table>

<!-- <table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
  <tr>
    <td align="center" valign="middle">
      <video muted autoplay width="100%">
        <source src="./video/overview.mov"  type="video/mp4">
      </video>
    </td>
  </tr>
  </tbody>
</table> -->

<table align=center width=800px>
    <tr>
        <td>
            <p align="justify" width="20%">
                We introduce MUTEX, a unified policy that learns to perform tasks based on task specifications in multiple modalities: image, video, text, and speech, in the form of instructions and goal descriptions. Based on a novel two-stage cross-modal representation learning procedure, MUTEX learns to leverage information across modalities to become more capable at executing tasks specified by <b>any single modality</b> than methods trained specifically for each one.
            </p>
        </td>
    </tr>
</table>


<br>
    <br>
        <hr>
            <h1 align="center">MUTEX Architecture</h1>
            <!-- <h2 align="center"></h2> -->
            <table border="0" cellspacing="10" cellpadding="0" align="center">
                <tbody>
                    <tr>
                        <td align="center" valign="middle">
                            <!-- add a video with thumbnail here -->
                            <video muted controls poster="src/architecture/Mutex_Architecture_thumbnail.png" width="100%">
                                <source src="src/architecture/Mutex_Architecture.mp4"  type="video/mp4">
                            </video>
                        </td>
                    </tr>
                </tbody>
            </table>
            <table border="0" cellspacing="10" cellpadding="0" align="center">
                <tr>
                    <td>
                        <p>
                            Task specifications in each modality are encoded with pretrained modality-specific encoders. During the first stage of training, one or more of these modalities are randomly selected and masked before being passed to projection layers and input to Mutex's transformer encoder. The resultant tokens are combined with observation tokens through N blocks of self- and cross-attention layers. The hidden state of the encoder is passed to Mutex's transformer encoder that is queried for actions (behavior learning loss) and the masked features and tokens (masked modeling loss), promoting action-specific cross-modal learning. In the second stage of training, all modalities are enriched with information from video features through a cross-modal matching loss. At test time, observations and a task specification of one modality are provided, based on which Mutex predicts closed-loop actions to achieve the task.
                        </p>
                    </td>
                </tr>
            </table>
<br>
<hr>
<h1 align="center">Simulation and Real-World Results</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody>
        <tr>
            <td align="center" valign="left">
                <a href="./src/resul_sim.png"> <img src="./src/result_sim.png" style="width:100%;"> </a>
            </td>
            <td align="center" valign="right">
                <a href="./src/resul_rw.png"> <img src="./src/result_rw.png" style="width:100%;"> </a>
            </td>
        </tr>
    </tbody>
</table>
<table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody><tr><td>
        <p align="justify" width="20%">
            We demonstrate that a policy that can execute task specified by multiple modalities generalizes better than a policy that can only execute tasks specified by any-single modality. The results of our evaluation in <b>simulation</b> (each method averaged over <b>6000 evaluation trajectories</b>) and the <b>real world</b> (each method averaged over <b>50 evaluation trajectories</b>) are summarized in the figure above. In both cases, we observe a significant improvement from using our unified policy MUTEX compared to modality-specific models indicating that the cross-modal representation learning procedure from MUTEX is able to leverage more information from other modalities.
        </p>
    </td></tr></tbody>
</table>

<br><hr>
<h1 align="center">Real World Demonstration</h1>
<table border="0" cellspacing="10" cellpadding="0" align="center">
    <td align="center" valign="middle">
        <!-- add the 6 vidoes in the directory src/rw_demo with title with play on click-->
        <!-- The six videos must be arranged in 2x3 grid -->
        <table>
          <tr>
            <td>
                <video width="380" height="213.75" controls poster="src/thumbnails/vid_thumbnail.png" style="margin-right: 10px; border: 0px solid #008BC6;">
                  <source src="src/rw_demo/video_demo_vid_reduced.mp4" type="video/mp4">
                </video>
                <p style="text-align: center; color: #008BC6;">Human Video Demonstration</p>
            </td>
            <td>
              <video width="380" height="213.75" controls poster="src/thumbnails/inst_thumbnail.png" style="margin-right: 10px;">
                <source src="src/rw_demo/text_inst_vid_reduced.mp4" type="video/mp4">
              </video>
              <p style="text-align: center; color: #275D54;">Text Instructions</p>
            </td>
          </tr>
          <tr>
            <td>
              <video width="380" height="213.75" controls poster="src/thumbnails/gl_thumbnail.png" style="margin-right: 10px;">
                <source src="src/rw_demo/text_goal_vid_reduced.mp4" type="video/mp4">
              </video>
              <p style="text-align: center; color: #DF9039;">Text Goal</p>
            </td>
            <td>
              <video width="380" height="213.75" controls poster="src/thumbnails/ai_thumbnail.png" style="margin-right: 10px;">
                <source src="src/rw_demo/speech_inst_vid_reduced.mp4" type="video/mp4">
              </video>
              <p style="text-align: center; color: #425C6B;">Speech Instructions</p>
            </td>
          </tr>
          <tr>
            <td>
              <video width="380" height="213.75" controls poster="src/thumbnails/ag_thumbnail.png">
                <source src="src/rw_demo/speech_goal_vid_reduced.mp4" type="video/mp4">
              </video>
              <p style="text-align: center; color: #838CD8;">Speech Goal</p>
            </td>
            <td>
              <video width="380" height="213.75" controls poster="src/thumbnails/img_thumbnail.png">
                <source src="src/rw_demo/image_goal_vid_reduced.mp4" type="video/mp4">
              </video>
              <p style="text-align: center; color: #D15C46;">Image Goal</p>
            </td>
          </tr>
        </table>
    </td>
</table>
<table border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody><tr><td>
        <p align="justify" width="20%">
            MUTEX can execute tasks specified by humans in any of the six modalities: <span style="color: #008BC6;"><strong>human video demonstrations</strong></span>, <span style="color: #275D54;"><strong>text instructions</strong></span>, <span style="color: #DF9039;"><strong>text goals</strong></span>, <span style="color: #425C6B;"><strong>speech instructions</strong></span>, <span style="color: #838CD8;"><strong>speech goals</strong></span>, and <span style="color: #D15C46;"><strong>image goals</strong></span>. To demonstrate the capabilities and robustness of MUTEX, we qualitatively evaluate MUTEX in the real world when a human specifies the tasks live through one of the modalities.
        </p>
    </td></tr></tbody>
</table>
<hr>
  <div align="center">
    <p>[Press Reload to change the tasks]</p>
  </div>
  <h1 align="center">MUTEX Dataset Visualization <button onclick="location.reload();" style="background: none; border: none; cursor: pointer;"><img src="src/reload.png" alt="Reload Page" style="vertical-align: middle; width: 30px; height: 30px;"></button></h1>
    <!-- Add a text describing the dataset details-->
    <table border="0" cellspacing="10" cellpadding="0" align="center">
        <tbody><tr><td>
            <p align="justify" width="20%">
                    We provide a dataset comprising <strong>100 simulated tasks</strong> based on <a href="https://arxiv.org/abs/2306.03310">LIBERO-100</a> and <strong>50 real-world</strong> tasks, annotated with <strong>50</strong> and <strong>30</strong> demonstrations per task (row following Task Name), respectively. We annotate each task with <strong>eleven</strong> alternative task specifications in each of the <strong>six</strong> following modalities (rows from top to bottom): video demonstration, image goal, text goal, text instructions, speech goal, and speech instructions.
            </p>
        </td></tr></tbody>
    </table>
  <br>
  <table id="myTable" border="0" cellspacing="10" cellpadding="0" align="center">
    <tbody></tbody>
  </table>

  <script>
    const boxWidth = 128;
    const boxHeight = 100;
    // make a list that maps index to type of task specification
    function createTextBox(text, boxWidth, boxHeight, text_type) {
        const textBox = document.createElement('div');
        // add text font size
        textBox.style.width = boxWidth + "px";
        textBox.style.height = boxHeight + "px";
        textBox.style.border = "1px solid black";
        textBox.style.display = "flex";
        textBox.style.justifyContent = "center";
        textBox.style.alignItems = "center";
        textBox.style.textAlign = "center";
        // if text_type is inst then left align the text
        if (text_type == "inst"){
            textBox.style.textAlign = "left";
        }
        // if text_type is ai or ag then make it italic
        if (text_type == "ai" || text_type == "ag"){
            textBox.style.fontStyle = "italic";
            textBox.style.border = "0px solid black";
        }

        const textContent = document.createElement('p');
        textContent.innerText = text;
        textBox.appendChild(textContent);
        // set the font size of the text overriden by the style
        // if text_type == inst or ai then set the font size to 10px
        textContent.style.fontSize = "12px";
        if (text_type == "inst" || text_type == "ai")
            textContent.style.fontSize = "11px";
        if (text_type == "name")
            textContent.style.fontSize = "16px";

        // add padding of 4px
        textBox.style.padding = "8px";
        if (text_type == 'name')
            textBox.style.padding = "2px";
        return textBox;
    }
    function createImageBox(src, boxWidth, boxHeight) {
        var img = document.createElement('img');
        img.src = src;
        img.style.width = boxWidth; // Set the width to your desired value
        img.style.height = boxHeight; // Set the width to your desired value
        img.style.margin = '0 auto'; // Align the text box to the center horizontally
        return img
    }
    function grabLanguageFromFilename(x) {
        var language;
        if (x[0] === x[0].toUpperCase()) { // Equivalent to x[0].isupper() in Python
            if (x.includes("EVAL")) {
                language = x.substring(x.indexOf("EVAL") + 6).split("_").join(" ");
                console.assert(language[0] !== " ");
            } else if (x.startsWith("RW")) {
                language = x.substring(x.indexOf("RW") + 4).split("_").join(" ");
            } else if (x.includes("SCENE10")) {
                language = x.substring(x.indexOf("SCENE") + 8).split("_").join(" ");
            } else {
                language = x.substring(x.indexOf("SCENE") + 7).split("_").join(" ");
            }
        } else {
            language = x.split("_").join(" ");
        }
        return language;
    }
    // var taskData = ['RW6_open_the_air_fryer_and_put_the_bowl_with_hot_dogs_in_it', 'RW5_put_the_bread_on_oven_tray_and_push_it_in_the_oven', 'RW7_put_the_book_in_the_back_compartment_of_the_caddy']
    // var taskData = ['KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it', 'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket', 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it']
    var tasks_all = ['RW6_open_the_air_fryer_and_put_the_bowl_with_hot_dogs_in_it', 'RW5_put_the_bread_on_oven_tray_and_push_it_in_the_oven', 'RW7_put_the_book_in_the_back_compartment_of_the_caddy', 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it', 'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket', 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it']
    var taskData = tasks_all.sort(() => Math.random() - Math.random()).slice(0, 3);
    const taskTypes = ["name", "robot", "vid", "img", "gl", "inst", "ag", "ai"];
    var tableData = [
      ['Task Name', '', '', ''],
      ['Robot Demonstration', '', '', ''],
      ['Video\nDemonstration', '', '', ''],
      ['Image Goals', '', '', ''],
      ['Text Goals', 'Text Goals Key1', 'Text Goals Key2', 'Text Goals Key3'],
      ['Text\nInstructions', 'Text Instructions Key 1', 'Text Instructions Key 2', 'Text Instructions Key 3'],
      ['Speech Goals', 'Speech Goals 1', 'Speech Goals 2', 'Speech Goals 3'],
      ['Speech\nInstructions', 'Speech Instructions 1', 'Speech Instruction 2', 'Speech Instructions 3']
    ];
    var jsonData = {
      "RW6_open_the_air_fryer_and_put_the_bowl_with_hot_dogs_in_it": {
          "ag": "Position the hot dog container in the air fryer basket.",
          "ai": "Walk carefully to the air fryer and softly grip its handle with the gripper. Gently open it by pulling on the handle. Head to the hot dog bowl and pick it up with caution. Lastly, gently insert the hot dog bowl into the open air fryer.",
          "gl": "The bowl that houses hot dogs is carefully put in the air fryer basket.",
          "inst": "1. Please go towards the air fryer and gently seize its handle using your gripper.\n2. Slowly reveal the interior of the air fryer by pulling the handle outward.\n3. Head to the bowl filled with hot dogs and firmly but gently grasp it.\n4. Carefully place it inside the now-open air fryer."
      },
      "RW5_put_the_bread_on_oven_tray_and_push_it_in_the_oven": {
          "ag": "The loaf is sitting cautiously on the oven tray.",
          "ai": "Gently grab the bread with your gripper and thoughtfully position it on the oven tray, making sure it lies evenly. Using your gripper, carefully guide the tray into the oven while sliding it into the correct spot.",
          "gl": "The bread has been cautiously placed on the oven tray.",
          "inst": "1. Kindly employ your clamp to softly take hold of the bread.\n2. Delicately rest it on the oven pan, verifying that it is even on the surface.\n3. Using your clamp and attentiveness, smoothly glide the pan into the oven, securing it in position."
      },
      "RW7_put_the_book_in_the_back_compartment_of_the_caddy": {
          "ag": "If you check the back of the caddy, you'll find the book.",
          "ai": "Identify the book and open your gripper to the right width. Move your gripper to the book and grasp it firmly. Locate the back compartment of the caddy. Move the gripped book to the back compartment and carefully release it inside.",
          "gl": "The book can be found in the rear section of the caddy.",
          "inst": "1. Discover the targeted book and set your gripper to the appropriate amplitude.\n2. Smoothly guide your gripper in the direction of the book and snatch it firmly.\n3. Search for the back section of the caddy.\n4. Move the held book towards the back section and lightly deposit it inside."
      },
      'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it': {
            "ag": "The moka pot is on the stove, and it's heating up.",
            "ai": "Carefully hold the knob and switch it to the 'on' position for the burner. Look around to find the Moka pot. Grab it by the handle and lift it up. Place the Moka pot over the stove and lower it until it sits securely on the surface. Now, feel free to let go of the handle.",
            "gl": "The stove is switched on, and the moka pot has been placed on it.",
            "inst": "1. Please kindly take hold of the control knob and swivel it to the correct 'on' setting for the burner.\n2. Carefully approach the Moka pot’s location.\n3. Grasp its handle to pick it up from where it’s resting.\n4. Guide it toward the stove while holding it securely above, and then gently let it settle on the stove’s surface.\n5. Finally, withdraw your hand from holding the Moka Pot."
      },
      'LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket': {
            "ag": "Please find the alphabet soup and the cream cheese box in the basket.",
            "ai": "Please grab the alphabet soup and move it from its current spot. Stand close to the basket. Gently slide the alphabet soup package into the basket. Let go of the alphabet soup package. Now, pick up the cream cheese container and lift it from where it is. Move toward the basket. Slowly place the cream cheese container inside the basket. Release your grip on the cream cheese container.",
            "gl": "Kindly check the basket for the wordy soup and the cream cheese container.",
            "inst": "1. Kindly hold the alphabet soup and separate it from its existing location.\n2. Steer toward the basket.\n3. Deliberately lower the alphabet soup container within the basket.\n4. Loosen your grip from the alphabet soup container.\n5. Pick up the cream cheese box and remove it from its present position.\n6. Stand by the basket.\n7. Gently place the cream cheese box into the basket.\n8. Free your hand from the cream cheese box."
      },
      'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it': {
            "ag": "Put the off-white mug in the microwave and make sure the door is fully closed.",
            "ai": "Kindly grab the yellow and white mug in front of you. Please place it inside and close the door securely before releasing your grip.",
            "gl": "The pastel-yellow cup is placed in the microwave and the door is tightly closed.",
            "inst": "1. Kindly hold the yellow and white mug, removing it from the position it rests upon.\n2. Situate the mug on the designated area inside.\n3. Ensure the microwave door is fully shut and let go of the cup."
      }
    };
    var tableBody = document.querySelector('#myTable tbody');

    for (var i = 0; i < tableData.length; i++) {
      var row = document.createElement('tr');

      for (var j = 0; j < tableData[i].length; j++) {
        var cell = document.createElement('td');
        if (j == 0) {
            const container = document.createElement("div");
            container.style.display = "flex";
            container.style.flexDirection = "column"; // Added line

            // Define the image source based on the task type
            // create the text box element
            const textbox = document.createElement("div");
            const text = tableData[i][j];
            textbox.style.border = "none";
            textbox.style.justifyContent = "center";
            textbox.style.alignItems = "center";
            textbox.style.textAlign = "center";

            // Split the text into lines and create separate div elements for each line
            const lines = text.split("\n");
            lines.forEach(line => {
              const lineDiv = document.createElement("div");
              lineDiv.textContent = line;
              textbox.appendChild(lineDiv);
            });

            container.appendChild(textbox);
            // if taskType is name, then no image is needed
            if (taskTypes[i] != 'name') {
                // Create the image box
                const imageSource = 'src/icons/' + taskTypes[i] + '_icon.png';
                const img = createImageBox(imageSource, 0.4*boxWidth, 0.4*boxWidth);
                // Append the image box and textbox to the container
                container.appendChild(img);
            }

            // Append the container to the cell
            cell.appendChild(container);
        }
        else {
            // For j=1,2 just change the data. Everything else remains the same.
            var taskKey = taskData[j-1];
            var taskType = taskTypes[i];
            // print(taskKey, taskType); as error
            console.log(taskKey, taskType);

            // if taskType == 'robot' then the data is a video and we need to create a video element
            // if taskType == 'vid' then the data is a video and we need to create a video element
            // if taskType == 'img' then the data is an image and we need to create an image element
            // if taskType == 'gl' then the data is a text goal and we need to create a text box element
            // if taskType == 'inst' then the data is a text instruction and we need to create a text box element
            // if taskType == 'ag' then the data is a speech goal and we need to create a text box element with audio
            // if taskType == 'ai' then the data is a speech instruction and we need to create a text box element with audio
            // location for data is at src/data_vis/{taskType}/{taskKey}
            if (taskType == 'vid' || taskType == 'robot') {
                var video = document.createElement('video');
                video.src = 'src/data_vis/' + taskType + '/' + taskType + '_' + taskKey;
                video.src += '.mp4';
                // print(video.src);
                // log as an error to print it in terminal
                console.error(video.src);
                video.style.display = 'block'; // Makes the video a block element
                video.style.width = boxWidth; // Set the width to your desired value
                video.style.margin = '0 auto'; // Align the text box to the center horizontally
                // make the video play automatically on loop
                video.autoplay = true;
                video.loop = true;
                cell.appendChild(video);
            }
            else if (taskType == 'img') {
                var img = document.createElement('img');
                img.src = 'src/data_vis/' + taskType + '/img_' + taskKey + '.png';
                img.style.width = boxWidth; // Set the width to your desired value
                // img.style.height = boxHeight; // Set the width to your desired value
                img.style.margin = '0 auto'; // Align the text box to the center horizontally
                cell.appendChild(img);
            }
            else if (taskType == 'gl') {

                tableData[i][j] = jsonData[taskKey][taskType];
                var textBox = createTextBox(tableData[i][j], boxWidth+68, boxHeight-40, text_type=taskType);
                textBox.style.margin = '0 auto'; // Align the text box to the center horizontally
                cell.appendChild(textBox);
            }
            else if (taskType == 'inst') {
                var container = document.createElement('div');
                container.style.position = 'relative';

                tableData[i][j] = jsonData[taskKey][taskType];
                var textBox = createTextBox(tableData[i][j], boxWidth+68, boxHeight+140, text_type=taskType);
                textBox.style.margin = '0 auto'; // Align the text box to the center horizontally
                container.appendChild(textBox);
                cell.appendChild(container);
            }
            else if (taskType == 'ag' || taskType == 'ai') {
                var container = document.createElement('div');
                container.style.position = 'relative';

                tableData[i][j] = jsonData[taskKey][taskType];
                if (taskType == 'ag') {
                    var textBox = createTextBox(tableData[i][j], boxWidth+68, boxHeight-60, text_type=taskType);
                }
                else {
                    var textBox = createTextBox(tableData[i][j], boxWidth+68, boxHeight+58, text_type=taskType);
                }
                textBox.style.margin = '0 auto'; // Align the text box to the center horizontally
                container.appendChild(textBox);
                // add some blank space between the text box and the audio box
                container.appendChild(document.createElement('br'));
                container.appendChild(document.createElement('br'));

                // add audio to the text box with the audio file
                var audio = document.createElement('audio');
                audio.src = 'src/data_vis/' + taskType + '/' + taskKey + '_' + taskType + '.mp3';
                audio.controls = true;
                // make the audio box width the same as the text box
                audio.style.width = textBox.style.width;
                container.appendChild(audio);

                // Position the audio element at the bottom center of the container
                audio.style.position = 'absolute';
                audio.style.bottom = '0';
                audio.style.left = '50%';
                // add some space to the audio element from the text box
                audio.style.transform = 'translateX(-50%)';

                cell.appendChild(container);
            }
            else if (taskType == 'name') {
                // add the name of the task
                // task_name is found by this python code: ' '.join(taskKey.split('_')[1:])
                // convert it to javascript
                var task_name = grabLanguageFromFilename(taskKey);
                var textBox = createTextBox(task_name, boxWidth+68, boxHeight-40, text_type=taskType);
                textBox.style.margin = '0 auto'; // Align the text box to the center horizontally
                // remove the border from the text box
                textBox.style.border = 'none';
                // make the text box background ligth grey
                // textBox.style.backgroundColor = '#f2f2f2';
                // amake the text box font size larger than x-large and text box fit the text
                textBox.style.fontSize = 'xx-large';
                textBox.style.width = 'fit-content';
                // make text content centered from top to bottom
                textBox.style.display = 'flex';

                cell.appendChild(textBox);
            }
        }
        row.appendChild(cell);
      }

      tableBody.appendChild(row);
    }
  </script>

<br>
<hr>
<!-- <center><h1>Citation</h1></center>

<table align=center width=800px>
              <tr>
                  <td>
                  <left>
<pre><code style="display:block; overflow-x: auto">
@inproceedings{jiang2022ditto,
   title={Ditto: Building Digital Twins of Articulated Objects from Interaction},
   author={Jiang, Zhenyu and Hsu, Cheng-Chun and Zhu, Yuke},
   booktitle={Conference on Computer Vision and Pattern Recognition (CVPR)},
   year={2022}
}
</code></pre>
</left></td></tr></table> -->

<!-- <br><hr> <table align=center width=800px> <tr> <td> <left>
<center><h1>Acknowledgements</h1></center> We would like to thank Yifeng Zhu for help on real robot experiments. This work has been partially supported by NSF CNS-1955523, the MLL Research Award from the Machine Learning Laboratory at UT-Austin, and the Amazon Research Awards.
 -->

<!-- </left></td></tr></table>
<br><br> -->

<div style="display:none">
<!-- Global site tag (gtag.js) - Google Analytics -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-PPXN40YS69"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());

  gtag('config', 'G-PPXN40YS69');
</script>
<!-- </center></div></body></div> -->


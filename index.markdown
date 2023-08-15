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
<center><h1><strong>MUTEX: Learning Unified Policies from Multimodal Task Specifications</strong></h1></center>
<center><h2>
    <a href="https://shahrutav.github.io/">Rutav Shah</a>&nbsp;&nbsp;&nbsp;
    <a href="https://robertomartinmartin.com/">Roberto Martín-Martín</a>&nbsp;&nbsp;&nbsp;
    <a href="https://cs.utexas.edu/~yukez">Yuke Zhu</a>&nbsp;&nbsp;&nbsp;
   </h2>
    <center><h2>
        <a href="https://www.cs.utexas.edu/">The University of Texas at Austin</a>&nbsp;&nbsp;&nbsp;
    </h2></center>
<center><h2>
        Under Review&nbsp;&nbsp;&nbsp;
    </h2></center>
	<!-- <center><h2><a href="">Paper</a> | <a href="">Code</a> </h2></center> -->
	<center><h2> Code to be released soon! </h2></center>


<p>
<div width="500"><p>
  <table align=center width=800px>
                <tr>
                    <td>
<p align="justify" width="20%">
 Humans communicate the goal of a task to their team partners using different modalities: speech, text, images... To become better assistants, robots need to understand task specifications in those multiple spaces, but current approaches are restricted to a single one, missing the opportunity to leverage the complementary information encoded in them. We present MUTEX, a novel approach to learning unified policies from multimodal task specifications that leverages cross-modal information thanks to its two-stage training procedure combining masked modeling and cross-modal matching, and its novel architecture based on transformers.  After training, MUTEX can follow a task specification in any of the six learned modalities (video demonstrations, goal images, text goal descriptions, text instructions, speech goal descriptions, and speech instructions), or combinations of specifications in several of them. We train and evaluate empirically the benefits of MUTEX in a novel dataset with 100 tasks in simulation and 50 tasks in the real world, annotated with multiple instances of task specifications in the six modalities, and observe improved performance over methods trained specifically for every single modality.
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
                We introduce MUTEX, a unified policy that learns to perform tasks based on task specifications in multiple modalities: text, image, video, and speech, including instructions and goal descriptions. Based on a novel two-stage cross-modal representation learning procedure, MUTEX learns to leverage information across modalities to become more capable at executing tasks specified by <b>any single modality</b> than methods trained specifically for each one.
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
                            <a href="./src/pipeline.png"> <img src="./src/pipeline.png" style="width:100%;"> </a>
                        </td>
                    </tr>
                </tbody>
            </table>
    <table width=800px>
        <tr>
            <td>
                <p align="justify" width="20%">
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
            We demonstrate that a policy that can execute task specified by multiple modalities generalizes better than a policy that can only execute tasks specified by any-single modality. The results of our evaluation in <b>simulation</b> (each method averaged over <b>6000 evaluation trajectories</b>) and the <b>real world</b> (each method averaged over <b>50 evaluation trajectories</b>) are summarized below. In both cases, we observe a significant improvement from using our unified policy MUTEX compared to modality-specific models indicating that the cross-modal representation learning procedure from MUTEX is able to leverage more information from other modalities.
        </p>
    </td></tr></tbody>
</table>

<!-- <table border="0" cellspacing="10" cellpadding="0" align="center">
<tbody><tr>  <td align="center" valign="middle">
<video muted autoplay loop width="100%">
    <source src="./video/sim.mp4"  type="video/mp4">
</video>
</td>
</tr>

</tbody>
</table>-->

<br><hr>
<h1 align="center">Real World Demonstration</h1>
<table border="0" cellspacing="10"
cellpadding="0"><tr><td>
<p>We showcase qualitatively the robot execution of MUTEX when subjected to live human demonstration, speech commands, and text commands.</p>
</td></tr></table>

<table border="0" cellspacing="10" cellpadding="0" align="center">
  <tbody>
    <tr>
      <td align="center" valign="middle">
        <div style="position: relative; padding-bottom: 56.25%; padding-top: 30px; height: 0;">
          <!-- <iframe src="https://www.youtube.com/embed/4T7HwLGNiuw" frameborder="0" allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe> -->
            <iframe width="560" height="315" src="https://www.youtube.com/embed/IiTXBH2vUxc" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
        </div>
      </td>
    </tr>
  </tbody>
</table>


<br><hr>
<h1 align="center">MUTEX Dataset Visualization</h1>
  <table id="myTable">
    <thead>
      <tr>
        <th></th>
        <th>Task 1</th>
        <th>Task 2</th>
      </tr>
    </thead>
    <tbody></tbody>
  </table>

  <!-- <script>
    const boxWidth = 100;
    const boxHeight = 100;

    function createTextBox(text, boxWidth, boxHeight) {
      const textBox = document.createElement('div');
      textBox.style.width = boxWidth + "px";
      textBox.style.height = boxHeight + "px";
      textBox.style.border = "1px solid black";
      textBox.style.display = "flex";
      textBox.style.justifyContent = "center";
      textBox.style.alignItems = "center";
      textBox.style.textAlign = "center";

      const textContent = document.createElement('p');
      textContent.innerText = text;
      textBox.appendChild(textContent);

      return textBox;
    }
    var tableData = [
      ['Image 1', 'Video 1', 'vipnsdoivnsdoivnsdo\nivnsoidnvosidnvosdnopvsn'],
      ['Image 2', 'Video 2', 'Text 2'],
      ['Image 3', 'Video 3', 'Text 3'],
      ['Image 4', 'Video 4', 'Text 4'],
      ['Image 5', 'Video 5', 'Text 5']
    ];

    var tableBody = document.querySelector('#myTable tbody');

    for (var i = 0; i < tableData.length; i++) {
      var row = document.createElement('tr');

      for (var j = 0; j < tableData[i].length; j++) {
        var cell = document.createElement('td');

        // You can add specific content based on the data type (image, video, audio, or text)
        if (j === 0) {

          var img = document.createElement('img');
          img.src = tableData[i][j];
          img.style.width = boxWidth; // Set the width to your desired value
          img.style.margin = '0 auto'; // Align the text box to the center horizontally
          cell.appendChild(img);
        } else if (j === 1) {
          var video = document.createElement('video');
          video.src = tableData[i][j];
          video.style.width = boxWidth; // Set the width to your desired value
          video.style.margin = '0 auto'; // Align the text box to the center horizontally
          cell.appendChild(video);
        } else if (j === 2) {
          var textBox = createTextBox(tableData[i][j], boxWidth, boxHeight);
          textBox.style.margin = '0 auto'; // Align the text box to the center horizontally
          cell.appendChild(textBox);
        }

        row.appendChild(cell);
      }

      tableBody.appendChild(row);
    }
  </script> -->

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


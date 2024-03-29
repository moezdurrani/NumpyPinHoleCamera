# NumpyPinHoleCamera

The initial version of my code included custom functions and objects that became obstacles when attempting to optimize with libraries such as Dask or NumPy. To address this, I rewrote the entire codebase, ensuring it utilizes only NumPy arrays, thereby eliminating any custom elements.
[Here](https://github.com/moezdurrani/ChromaticAberration) is the link to the previous versions of the code.

A video of the program running can be found [here](https://drive.google.com/file/d/1V2--q_knE8TUE4qh9Bu2tR653AiOJLmq/view?usp=sharing).

Furthermore, I have made modifications to enable running the code directly from the terminal with inputs like the following:

```
python3 main.py \
  --frontImg "tree.jpg" \
  --modelName "objFiles/prism.obj" \
  --maxDepth 4 \
  --modelx 0 \
  --modely 0 \
  --modelz -18 \
  --modelxR 90 \
  --modelyR 45 \
  --modelzR 0 \
  --imgWidth 1024 \
  --imgHeight 768 \
  --zoomFactor 3
```

<ul>
<li>The --frontImg parameter specifies the background image file name</li>
<li>The --modelName parameter identifies the 3D model to be rendered and its location within the main code folder</li>
<li>The --maxDepth parameter determines the maximum depth for ray tracing recursion, allowing for a specific number of reflections within the scene</li>
<li>The modelx, modely, and modelz parameters set the 3D coordinates for the model's position within the scene, while modelxR, modelyR, and modelzR control the model's rotation around the x, y, and z axes, respectively</li>
<li>The --imgWidth and --imgHeight inputs define the output image's resolution</li>
<li>The --zoomFactor adjusts the background image scaling, effectively zooming in or out to fit the scene. Increasing the zoom factor zooms out, offering a broader view of the background image</li>
</ul>

Below are some examples of rendered images.

<h3>Without any object in the scene</h3>

<table align="center">
  <tr>
    <th>Zoom Factor 1</th>
    <th>Zoom Factor 2</th>
  </tr>
  <tr>
    <td><img src="https://github.com/moezdurrani/NumpyPinHoleCamera/blob/main/images/Zoom1.png" alt="gray cube"></td>
    <td><img src="https://github.com/moezdurrani/NumpyPinHoleCamera/blob/main/images/Zoom2.png" alt="gray cube"></td>
  </tr>
<tr>
    <th>Zoom Factor 3</th>
  </tr>
<tr>
    <td><img src="https://github.com/moezdurrani/NumpyPinHoleCamera/blob/main/images/Zoom3.png" alt="gray cube"></td>
  </tr>
</table>

<h3>With a glass sphere in the scene</h3>

<table align="center">
  <tr>
    <th>Zoom Factor 1</th>
    <th>Zoom Factor 2</th>
  </tr>
  <tr>
    <td><img src="https://github.com/moezdurrani/NumpyPinHoleCamera/blob/main/images/Spherezoom1.png" alt="gray cube"></td>
    <td><img src="https://github.com/moezdurrani/NumpyPinHoleCamera/blob/main/images/Spherezoom2.png" alt="gray cube"></td>
  </tr>
<tr>
    <th>Zoom Factor 3</th>
  </tr>
<tr>
    <td><img src="https://github.com/moezdurrani/NumpyPinHoleCamera/blob/main/images/Spherezoom3.png" alt="gray cube"></td>
  </tr>
</table>
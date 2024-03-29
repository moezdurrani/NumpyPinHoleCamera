# NumpyPinHoleCamera

The initial version of my code included custom functions and objects that became obstacles when attempting to optimize with libraries such as Dask or NumPy. To address this, I rewrote the entire codebase, ensuring it utilizes only NumPy arrays, thereby eliminating any custom elements.
[Here](https://github.com/moezdurrani/ChromaticAberration?tab=readme-ov-file) is the link to the previous versions of the code.

Additionally, the code was modified to be able to run from the terminal using the following input

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

The "frontImg" input takes the name of the image file that is going to be in the background. The "modelName" input takes the name of the 3d model that needs to be rendered and its location inside the main folder of the code.
The "maxDepth" parameter specifies the maximum depth for ray tracing recursion, allowing for a certain number of reflections within the rendered scene. The parameters "modelx", "modely", and "modelz" define the 3D coordinates for the position of the model within the scene. These are followed by "modelxR", "modelyR", and "modelzR", which determine the rotation of the model around the x, y, and z axes, respectively, providing control over the orientation of the model.
The "imgWidth" and "imgHeight" inputs set the resolution of the output image, dictating how many pixels wide and tall the final image will be. Lastly, the "zoomFactor" input allows for scaling of the background image, effectively zooming in or out to adjust how the background image fits within the scene. Increasing the zoomfactore will zoom out of the image that is in the background. 


# Google Vision Entity Extraction
This is the python code to generate entities from the image files. To use this code, you need a Google Vision account and a billing account. The files have been generated and are available from the OneDrive. Files are .json files storing the response from the Google Vision API.

We may not use these in the project; however, they are available if we want to use them to enhance the multimodal tasks.


## Model Files
Model files are too large to add to repository, stored on OneDrive link (PDF on main) under Googe Vision Entity Extraction.

### Face Detection
Face Detection detects multiple faces within an image along with the associated key facial attributes such as emotional state or wearing headwear. The Response output includes the bounding boxes/polygon, angle, tilt, confidence scores for emotion and headwear likelihood.

See example response: https://cloud.google.com/vision/docs/detecting-faces

### Web Detection
Web Detection detects Web references to an image. Entities are linked to an ID which may be from the Google KG but this is unclear. The API also detects pages with matching images, other matching images, visually similar images and full matching images, providing a URL for these. The API also attempts to label the image itself.

See example response: https://cloud.google.com/vision/docs/detecting-web

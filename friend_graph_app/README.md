# Friend Recommendation with Graph Neural Networks (web demo)

A small Flask app with two pages:
- **Network Explorer**: an interactive graph of about 150 users. Color nodes by age group or country,
  click a person to see their profile and friends, and click a friend's name to walk through the network.
  A yellow halo means that friend also knows other friends of the person you clicked.
- **Results**: your baseline comparison, charts, limitations, and ideas to improve Precision@K.

## Setup

1. Put this whole `friend_graph_app` folder inside your project folder, next to
   `SocialMediaUsersDataset_v2.csv` and `friendship_edges.csv`
   (or copy the two CSVs into this folder). The app only reads them and never changes them.
2. Install the two libraries it needs (no PyTorch needed). Use the same Python that runs your other scripts:

       python -m pip install flask pandas

   If you have more than one Python, use the full path:

       & "C:\Users\hoody\AppData\Local\Python\pythoncore-3.13-64\python.exe" -m pip install flask pandas

3. Run it from inside this folder:

       python app.py

4. Open http://127.0.0.1:5000 in your browser. It works offline because the graph library is bundled.

## Notes
- The first start takes a few seconds while it loads the 100,000 users.
- The starting sample is fixed (seed 42), so the demo looks the same every time. The "Good places to start"
  list in the side panel names people whose friends know each other, so halos show up right away.
- Change `SAMPLE_SIZE` at the top of `app.py` to show more or fewer people.
- The numbers on the Results page are typed into `templates/results.html`. If you rerun your experiments, update them there.

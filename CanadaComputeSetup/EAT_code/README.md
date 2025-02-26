# Local Dependencies

In addition to the requirements listed in `requirements.txt`, some packages must be downloaded manually and installed from local when submitting the job.

Here's the list:

- dlib==19.24.6
- imageio_ffmpeg==0.6.0 (there is a bug/breaking change at 0.4.2, so theoretically >0.4.2 works here)

It helps to speed up the process if the wheels are pre-built from source using `pip wheel <package.tar.gz>`.

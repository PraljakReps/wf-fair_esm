FROM 812206152185.dkr.ecr.us-west-2.amazonaws.com/wf-base:fbe8-main

# Or use managed library distributions through the container OS's package
# manager.
RUN apt-get update -y

# ESM dependencies...
#RUN conda install pip \
#    && pip install torch fair-esm
RUN pip install torch fair-esm fsspec

# run low-dimensional algorithms
RUN pip install -U scikit-learn
# plot figures
RUN pip install matplotlib

COPY wf /root/wf

# STOP HERE:
# The following lines are needed to ensure your build environement works
# correctly with latch.
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
RUN  sed -i 's/latch/wf/g' flytekit.config
RUN python3 -m pip install --upgrade latch
WORKDIR /root

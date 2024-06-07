FROM neurips23

RUN apt update && apt install -y wget swig
# RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-0-Linux-x86_64.sh
RUN wget https://mirrors.aliyun.com/anaconda/archive/Anaconda3-2023.03-0-Linux-x86_64.sh
RUN bash Anaconda3-2023.03-0-Linux-x86_64.sh -b

ENV PATH /root/anaconda3/bin:$PATH
ENV CONDA_PREFIX /root/anaconda3/

RUN conda install -c pytorch faiss-cpu
COPY install/requirements_conda.txt ./
# conda doesn't like some of our packages, use pip 
RUN python3 -m pip install -r requirements_conda.txt
RUN pip install memory_profiler

COPY neurips23/filter/faiss/bow_id_selector.swig ./
RUN mkdir python_bindings
COPY neurips23/filter/hnsw/__init__.py ./python_bindings/
COPY neurips23/filter/hnsw/bindings.cpp ./python_bindings/
COPY neurips23/filter/hnsw/LazyIndex.py ./python_bindings/
COPY neurips23/filter/hnsw/setup.py ./python_bindings/
COPY neurips23/filter/hnsw/bruteforce.h ./python_bindings/
COPY neurips23/filter/hnsw/hnswalg.h ./python_bindings/
COPY neurips23/filter/hnsw/hnswlib.h ./python_bindings/
COPY neurips23/filter/hnsw/space_ip.h ./python_bindings/
COPY neurips23/filter/hnsw/space_l2.h ./python_bindings/
COPY neurips23/filter/hnsw/visited_list_pool.h ./python_bindings/

RUN pip uninstall hnswlib
RUN python3 python_bindings/setup.py bdist_wheel
RUN pip install dist/*

RUN mkdir uint8_knn_bindings
COPY neurips23/filter/uint8_knn_file/bindings.cc ./uint8_knn_bindings/
COPY neurips23/filter/uint8_knn_file/setup.py ./uint8_knn_bindings/
COPY neurips23/filter/uint8_knn_file/test.py ./uint8_knn_bindings/
COPY neurips23/filter/uint8_knn_file/uint8_knn.hpp ./uint8_knn_bindings/
RUN rm -rf build
RUN rm -rf dist
RUN python3 uint8_knn_bindings/setup.py bdist_wheel
RUN pip uninstall uint8_knn -y
RUN pip install dist/*

RUN swig -c++ -python -I$CONDA_PREFIX/include -Ifaiss bow_id_selector.swig
RUN g++ -shared  -O3 -g -fPIC bow_id_selector_wrap.cxx -o _bow_id_selector.so  \
      -I $( python -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())" )   \
      -I $CONDA_PREFIX/include $CONDA_PREFIX/lib/libfaiss_avx2.so -Ifaiss

RUN python3 -c 'import faiss; print(faiss.IndexFlatL2); print(faiss.__version__)'

rm -rf build
python setup.py bdist_wheel
pip uninstall uint8_knn -y
pip install dist/*
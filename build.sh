echo "\n\n====================== START TO BUILD AND INSTALL TENSORFLOW ====================\n"
rm -rf /tmp/*
echo "\n\n######### START TO BUILD #################\n\n"
bazel build -c opt --jobs=32 --verbose_failures //tensorflow/tools/pip_package:build_pip_package
echo "\n\n######### START TO PACKAGE #################\n\n"
./bazel-bin/tensorflow/tools/pip_package/build_pip_package /home/rui/localTool/tensorflow_pkg
pip uninstall -y tensorflow
echo "\n\n######### START TO INSTALL #################\n\n"
pip install ~/localTool/tensorflow_pkg/tensorflow-2.16.1-cp311-cp311-linux_x86_64.whl
print "\n\n========================= BUILD AND INSTALLATION DONE! ==========================\n"

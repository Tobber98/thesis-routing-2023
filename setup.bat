cd .\rust\routing
cargo b -r
cd ..\o-routing
cargo b -r
cd ..\nd-routing
cargo b -r
cd ..\..\python
pip install -r requirements.txt
cd ..
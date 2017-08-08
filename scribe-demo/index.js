const PythonShell = require('python-shell')
const express = require('express')
const bodyParser = require('body-parser')
const fs = require('fs')
const cors = require('cors')
const app = express()


// Tell express to use the body-parser middleware and to not parse extended bodies
app.use(bodyParser.json())

//Allow CORS
app.use(cors())
 
// Route that receives a POST request to /sms
app.post('/handwritingDL', function (req, res) {
	const body = req.body
	console.log(body)
	var options = {
        mode: 'text',
        scriptPath: '../',
        args: ['--sample', '--text', body.text ,'--save_path', '../saved/model.ckpt', '--bias', body.bias, '--style', body.style, '--data_dir', '../data', '--no_info']
	};
	var pyShell = new PythonShell('run.py',options, function(err){
	        if(err) throw err;
	});
	pyShell.end(function(err){
		var filePath = './logs/figures/'+body.text+'.png';
		var img = fs.readFileSync(filePath);
		//res.writeHead(200, {'Content-Type': 'application/json' });
		res.json(new Buffer(img).toString('base64'));
		fs.unlinkSync(filePath);
	});
	
})
 
// Tell our app to listen on port 3000
app.listen(3000, function (err) {
  if (err) {
    throw err
  }
 
  console.log('Server started on port 3000')
})

var myPythonScriptPath = 'run.py';

// Use python shell
var PythonShell = require('python-shell');

console.log("Please Input a String")
var stdin = process.openStdin();
stdin.addListener("data",function(d){
	d = d.tostring().replace('\n', '');
	var options = {
		mode : 'text',
		pythonPath: '/usr/bin/python2.7',
		scriptPath: __dirname + '/../',
		args: ['--sample',' --text ' + "'" + d + "'"]
	}
	console.log(options.scriptPath)
	// var pyshell = new PythonShell(myPythonScriptPath);
	PythonShell.run(myPythonScriptPath,options,function(err){
		if(err)throw err;
	});
	
	// pyshell.on('message', function (message) {

	//     // received a message sent from the Python script (a simple "print" statement)
	//     console.log(message);
	// });

// end the input stream and allow the process to exit
	// pyshell.end(function (err) {
	//     if (err){
	//         throw err;
	//     };

	//     console.log('finished');
	// });
});


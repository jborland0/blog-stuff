import React from 'react';
import logo from './logo.svg';
import './App.css';
import 'bootstrap/dist/css/bootstrap.min.css';
import Navbar from 'react-bootstrap/Navbar';
import Moodilyzer from './components/Moodilyzer';
import config from './config/index';

function App() {
  return (
	<>
	  <Navbar bg="dark" variant="dark">
		<Navbar.Brand href="/">
		  <img
			alt=""
			src={require('./mosquito_tiny.png')}
			className="d-inline-block"
			style={{marginTop: -7}}
		  />{' '}
		  Moodilyzer
		</Navbar.Brand>
	  </Navbar>
	  <div style={{height: '10px', width: '100%'}}></div>
	  <Moodilyzer config={config}/>
	</>
  );
}

export default App;

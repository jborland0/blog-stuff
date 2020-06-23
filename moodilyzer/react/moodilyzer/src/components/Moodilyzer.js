import React from 'react';
import Form from 'react-bootstrap/Form';
import Button from 'react-bootstrap/Button';
import { Col, Container, Row } from "react-bootstrap";
import $ from 'jquery';

class Moodilyzer extends React.Component {
  	constructor(props) {
		super(props);
		
		this.state = {
			text: "",
			lstm: {
				grateful: 0,
				happy: 0,
				hopeful: 0,
				determined: 0,
				aware: 0,
				stable: 0,
				frustrated: 0,
				overwhelmed: 0,
				angry: 0,
				guilty: 0,
				lonely: 0,
				scared: 0,
				sad: 0		
			},
			elmo: {
				grateful: 0,
				happy: 0,
				hopeful: 0,
				determined: 0,
				aware: 0,
				stable: 0,
				frustrated: 0,
				overwhelmed: 0,
				angry: 0,
				guilty: 0,
				lonely: 0,
				scared: 0,
				sad: 0		
			},
		};
		
		this.timers = {
			lstm: null,
			elmo: null
		};
	}
	
  mergeState(newState) {
	  this.mergeObject(this.state, newState);
	  this.setState(newState);
  }
  
  mergeObject(oldObj, newObj) {
	  var self = this;
	  // for each key in the old object
	  Object.keys(oldObj).map(function(key) {
		  // if the key is not defined in the new object
		  if (newObj[key] === undefined) {
			  // add value to new object
			  newObj[key] = oldObj[key];
		  } else if (self.isObject(oldObj[key]) && self.isObject(newObj[key])) {
			  // call recursively on child objects
			  self.mergeObject(oldObj[key], newObj[key]);
		  }
	  });
  }
  
  isObject(o) {
	  return typeof o === 'object' && o !== null
  }
  
  getCookie(name) {
    var cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        var cookies = document.cookie.split(';');
        for (var i = 0; i < cookies.length; i++) {
            var cookie = $.trim(cookies[i]);
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
  }

  moodilyze(event) {
	var self = this;
	event.preventDefault();
	var csrftoken = this.getCookie('csrftoken');
	
	// start timer for lstm
	this.timers.lstm = window.setInterval(this.randomize, 300, this, 'lstm');

	$.ajax({
		type: "get",
		url: this.props.config.urls.lstm,
		cache: false,
		data: { text: this.state.text },
		headers: {
			'Accept': 'application/json',
			'Content-Type': 'application/json',
			'X-CSRFToken': csrftoken
		}
	}).done(function (data) {
		window.clearInterval(self.timers.lstm);
		self.mergeState({ lstm: data });
	}).fail(function (xhr) {
		window.clearInterval(self.timers.lstm);
		console.log(xhr.responseText);
	});
	
	// start timer for elmo
	this.timers.elmo = window.setInterval(this.randomize, 300, this, 'elmo');

	$.ajax({
		type: "get",
		url: this.props.config.urls.elmo,
		cache: false,
		data: { text: this.state.text },
		headers: {
			'Accept': 'application/json',
			'Content-Type': 'application/json',
			'X-CSRFToken': csrftoken
		}
	}).done(function (data) {
		window.clearInterval(self.timers.elmo);
		self.mergeState({ elmo: data });
	}).fail(function (xhr) {
		window.clearInterval(self.timers.elmo);
		console.log(xhr.responseText);
	});
  }
  
  randomPct() {
	  return Math.floor(Math.random() * Math.floor(101));
  }

  randomize(self, classifierName) {
	  var newState = {};
	  
	  newState[classifierName] = {
		grateful: self.randomPct(),
		happy: self.randomPct(),
		hopeful: self.randomPct(),
		determined: self.randomPct(),
		aware: self.randomPct(),
		stable: self.randomPct(),
		frustrated: self.randomPct(),
		overwhelmed: self.randomPct(),
		angry: self.randomPct(),
		guilty: self.randomPct(),
		lonely: self.randomPct(),
		scared: self.randomPct(),
		sad: self.randomPct()		  
	  };
	  
	  self.mergeState(newState);
  }
	
  render() {
    return (
	  <Container fluid>
	    <Row>
		  <Col sm={6}>
			<Form onSubmit={(event) => this.moodilyze(event)}>
			  <Form.Group controlId="text">
				<Form.Label>How are you today?</Form.Label>
				<Form.Control as="textarea" rows="9" value={this.state.text} onChange={(event) => this.mergeState({text: event.target.value})} />
			  </Form.Group>
			  <Button variant="primary" type="submit" className="float-right">
				Moodilyze!
			  </Button>
			</Form>
		  </Col>
		  <Col sm={6}>
		    <Container className="border">
			  <Row>
				<Col><strong>Mood</strong></Col>
				<Col><strong>LSTM</strong></Col>
				<Col><strong>ELMo</strong></Col>
			  </Row>
			  <Row>
			    <Col>Grateful</Col>
				<Col><Mood backgroundColor="#007984" width={this.state.lstm.grateful} /></Col>
				<Col><Mood backgroundColor="#007984" width={this.state.elmo.grateful} /></Col>
			  </Row>
			  <Row>
			    <Col>Happy</Col>
				<Col><Mood backgroundColor="#119f54" width={this.state.lstm.happy} /></Col>
				<Col><Mood backgroundColor="#119f54" width={this.state.elmo.happy} /></Col>
			  </Row>
			  <Row>
			    <Col>Hopeful</Col>
				<Col><Mood backgroundColor="#2bb650" width={this.state.lstm.hopeful} /></Col>
				<Col><Mood backgroundColor="#2bb650" width={this.state.elmo.hopeful} /></Col>
			  </Row>
			  <Row>
			    <Col>Determined</Col>
				<Col><Mood backgroundColor="#84c939" width={this.state.lstm.determined} /></Col>
				<Col><Mood backgroundColor="#84c939" width={this.state.elmo.determined} /></Col>
			  </Row>
			  <Row>
			    <Col>Aware</Col>
				<Col><Mood backgroundColor="#ffdd00" width={this.state.lstm.aware} /></Col>
				<Col><Mood backgroundColor="#ffdd00" width={this.state.elmo.aware} /></Col>
			  </Row>
			  <Row>
			    <Col>Stable</Col>
				<Col><Mood backgroundColor="#ffbb00" width={this.state.lstm.stable} /></Col>
				<Col><Mood backgroundColor="#ffbb00" width={this.state.elmo.stable} /></Col>
			  </Row>
			  <Row>
			    <Col>Frustrated</Col>
				<Col><Mood backgroundColor="#ff9100" width={this.state.lstm.frustrated} /></Col>
				<Col><Mood backgroundColor="#ff9100" width={this.state.elmo.frustrated} /></Col>
			  </Row>
			  <Row>
			    <Col>Overwhelmed</Col>
				<Col><Mood backgroundColor="#f84d34" width={this.state.lstm.overwhelmed} /></Col>
				<Col><Mood backgroundColor="#f84d34" width={this.state.elmo.overwhelmed} /></Col>
			  </Row>
			  <Row>
			    <Col>Angry</Col>
				<Col><Mood backgroundColor="#f01745" width={this.state.lstm.angry} /></Col>
				<Col><Mood backgroundColor="#f01745" width={this.state.elmo.angry} /></Col>
			  </Row>
			  <Row>
			    <Col>Guilty</Col>
				<Col><Mood backgroundColor="#d23ca1" width={this.state.lstm.guilty} /></Col>
				<Col><Mood backgroundColor="#d23ca1" width={this.state.elmo.guilty} /></Col>
			  </Row>
			  <Row>
			    <Col>Lonely</Col>
				<Col><Mood backgroundColor="#6b5bbc" width={this.state.lstm.lonely} /></Col>
				<Col><Mood backgroundColor="#6b5bbc" width={this.state.elmo.lonely} /></Col>
			  </Row>
			  <Row>
			    <Col>Scared</Col>
				<Col><Mood backgroundColor="#6ba8f1" width={this.state.lstm.scared} /></Col>
				<Col><Mood backgroundColor="#6ba8f1" width={this.state.elmo.scared} /></Col>
			  </Row>
			  <Row>
			    <Col>Sad</Col>
				<Col><Mood backgroundColor="#00b7e1" width={this.state.lstm.sad} /></Col>
				<Col><Mood backgroundColor="#00b7e1" width={this.state.elmo.sad} /></Col>
			  </Row>
			</Container>
		  </Col>
	    </Row>
	  </Container>
	);
  }
}

class Mood extends React.Component {
	render() {
	  return(
		<div style={{backgroundColor: this.props.backgroundColor, height: '20px', width: this.props.width}}></div>
	  );
	}
}

export default Moodilyzer;
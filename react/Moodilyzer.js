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
		};
	}
	
  mergeState(newState) {
	  var self = this;
	  // for each key in the current state
	  Object.keys(this.state).map(function(key) {
			// add it to the new state if it is not present
			if (newState[key] === undefined) newState[key] = self.state[key];
	  });
	  this.setState(newState);
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
	console.log("calling ajax");

	$.ajax({
		type: "get",
		url: "/moodilyzer/nn/jsonresults/",
		cache: false,
		data: { text: this.state.text },
		headers: {
			'Accept': 'application/json',
			'Content-Type': 'application/json',
			'X-CSRFToken': csrftoken
		}
	}).done(function (data) {
		console.log("success!");
		self.mergeState(data);
	}).fail(function (xhr) {
		console.log(xhr.responseText);
	}).always(function () {
		console.log("something happened.");
	});
	console.log("called ajax.");
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
			    <Col>Grateful</Col>
				<Col><Mood backgroundColor="#007984" width={this.state.grateful} /></Col>
			  </Row>
			  <Row>
			    <Col>Happy</Col>
				<Col><Mood backgroundColor="#119f54" width={this.state.happy} /></Col>
			  </Row>
			  <Row>
			    <Col>Hopeful</Col>
				<Col><Mood backgroundColor="#2bb650" width={this.state.hopeful} /></Col>
			  </Row>
			  <Row>
			    <Col>Determined</Col>
				<Col><Mood backgroundColor="#84c939" width={this.state.determined} /></Col>
			  </Row>
			  <Row>
			    <Col>Aware</Col>
				<Col><Mood backgroundColor="#ffdd00" width={this.state.aware} /></Col>
			  </Row>
			  <Row>
			    <Col>Stable</Col>
				<Col><Mood backgroundColor="#ffbb00" width={this.state.stable} /></Col>
			  </Row>
			  <Row>
			    <Col>Frustrated</Col>
				<Col><Mood backgroundColor="#ff9100" width={this.state.frustrated} /></Col>
			  </Row>
			  <Row>
			    <Col>Overwhelmed</Col>
				<Col><Mood backgroundColor="#f84d34" width={this.state.overwhelmed} /></Col>
			  </Row>
			  <Row>
			    <Col>Angry</Col>
				<Col><Mood backgroundColor="#f01745" width={this.state.angry} /></Col>
			  </Row>
			  <Row>
			    <Col>Guilty</Col>
				<Col><Mood backgroundColor="#d23ca1" width={this.state.guilty} /></Col>
			  </Row>
			  <Row>
			    <Col>Lonely</Col>
				<Col><Mood backgroundColor="#6b5bbc" width={this.state.lonely} /></Col>
			  </Row>
			  <Row>
			    <Col>Scared</Col>
				<Col><Mood backgroundColor="#6ba8f1" width={this.state.scared} /></Col>
			  </Row>
			  <Row>
			    <Col>Sad</Col>
				<Col><Mood backgroundColor="#00b7e1" width={this.state.sad} /></Col>
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

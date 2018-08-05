import React, { Component } from 'react';
import './App.css';
import ajax from './tools/ajax.js'

class App extends Component {
  constructor(props) {
    super(props);
    this.textInput = React.createRef();
    this.handelSubmit = this.handelSubmit.bind(this);

    this.state = {
      dialogs:[{'bot':'How can I help you?'}]
    }
  }

  handelSubmit() {
    var _this = this;
    var inputData = {
      "text": this.textInput.current.value
    }

    var newDialogs = this.state.dialogs;
    newDialogs.push({'user': this.textInput.current.value});
    this.setState({dialogs: newDialogs});
    this.textInput.current.value = "";
    var newDialogs = _this.state.dialogs;
    newDialogs.push({'bot': 'loading'});
    _this.setState({dialogs: newDialogs});
    ajax({
      method: 'POST',
      url: 'http://localhost:8000/api/chatterbot/',
      data:  JSON.stringify(inputData),
      contentType: 'application/json',
      success: function (response) {
        var text = JSON.parse(response).text;
        var newDialogs = _this.state.dialogs;
        newDialogs[newDialogs.length - 1].bot = text;
        _this.setState({dialogs: newDialogs});
      }
    });
  }
  render() {
    const listItems = this.state.dialogs.map((data, index) => {
      console.log(data.bot);
      if (data.bot) {
        if (data.bot != 'loading') {
          return <div key={index} className="list-item"><li className="list-bot-item">{data.bot}</li></div>;
        } else {
          return <div key={index} className="list-item"><li className="list-bot-item"><span className = "list-dot dot-first"></span><span className = "list-dot dot-second"></span><span className = "list-dot dot-third"></span></li></div>;
        }
      } else {
          return <div key={index} className="list-item"><li className="list-user-item"><span className="list-user-corner"></span>{data.user}</li></div>;
      }
    }  
    );
    return (
      <div className="container">
        <div className="jumbotron mt-1">
          <h2>ChatBot Example</h2>
          <hr className="my-2"/>

          <div className="row">
            <div className="col-xs-6 offset-xs-3">
              <ul className="list-group chat-log js-chat-log">
                {listItems}
              </ul>

              <div className="input-group input-group-lg mt-1">
                <input ref={this.textInput} type="text" className="form-control js-text" placeholder="Type something to begin..."/>
                <span className="input-group-btn">
                  <button onClick={this.handelSubmit} className="btn btn-primary js-say">Submit</button>
                </span>
              </div>   
            </div>
          </div>
        </div>
      </div>

    );
  }
}

export default App;

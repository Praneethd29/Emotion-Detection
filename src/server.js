const http = require('http');
const express = require('express');
const socketIo = require('socket.io');
const cors = require('cors');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
  cors: {
    origin: "http://localhost:3000",
    methods: ["GET", "POST"],
    credentials: true
  }
});

app.use(cors({
  origin: "http://localhost:3000",
  methods: ["GET", "POST"],
  credentials: true
}));

let waitingUsers = [];

io.on('connection', (socket) => {
  console.log(`New user connected: ${socket.id}`);

  socket.on('join', (userData) => {
    console.log(`User joined: ${socket.id}`, userData);

    if (!waitingUsers.some(user => user.socket.id === socket.id)) {
      waitingUsers.push({ socket, userData });
      console.log('Current waiting users:', waitingUsers.map(user => user.socket.id));
    }

    if (waitingUsers.length >= 2) {
      const [user1, user2] = waitingUsers.splice(0, 2);
      console.log(`Pairing users: ${user1.socket.id} and ${user2.socket.id}`);

      user1.socket.emit('paired', { partner: user2.socket.id });
      user2.socket.emit('paired', { partner: user1.socket.id });

      const room = user1.socket.id + '#' + user2.socket.id;
      user1.socket.join(room);
      user2.socket.join(room);

      user1.socket.on('send_message', (data) => {
        console.log(`Message from ${user1.socket.id}: ${data.text}`);
        io.to(room).emit('receive_message', data);
      });
      user2.socket.on('send_message', (data) => {
        console.log(`Message from ${user2.socket.id}: ${data.text}`);
        io.to(room).emit('receive_message', data);
      });

      user1.socket.on('disconnect', () => {
        console.log(`User disconnected: ${user1.socket.id}`);
        io.to(room).emit('partner_disconnected');
      });
      user2.socket.on('disconnect', () => {
        console.log(`User disconnected: ${user2.socket.id}`);
        io.to(room).emit('partner_disconnected');
      });
    }
  });

  socket.on('disconnect', () => {
    console.log(`User disconnected: ${socket.id}`);
    waitingUsers = waitingUsers.filter(user => user.socket.id !== socket.id);
    console.log('Current waiting users:', waitingUsers.map(user => user.socket.id));
  });
});

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
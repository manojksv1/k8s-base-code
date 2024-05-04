data "aws_ami" "nat_instances" {
  most_recent = true
  owners      = ["amazon"]
  filter {
    name   = "name"
    values = ["amzn-ami-vpc-nat-2018.03.0.20221018.0-x86_64-ebs"]
  }
}

resource "aws_instance" "nat_server" {
  count = var.nat_server_status ? 1:0
  ami           = data.aws_ami.nat_instances.id
  instance_type = var.nat-server-instance-type
  tags = {
    Name  = "nat-server"
    from  = "trrafrom"
    owner = "manojs"
  }
  source_dest_check = false
  subnet_id         = aws_subnet.public-subnet-test[0].id
  #security_groups   = [aws_security_group.nat-sg.id]
  vpc_security_group_ids = [aws_security_group.nat-sg.id]
}

resource "aws_security_group" "nat-sg" {
  name        = "nat-server-sg"
  description = "allow http and https traffic"
  vpc_id      = aws_vpc.vpc.id

  ingress {
    description = "HTTP ingress"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    description = "HTTPS ingress"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

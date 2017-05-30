function mail_notification(target, subject, msg)
%% EMAIL_NOTIFICATION
% Generate automatic email notification to target receiver, with subject
% and message content.
%
% This is mainly used to track working process on Matlab servers.

% Note that the email and password are created solely on purpose of email
% notification. Any user can use it.

mail = 'ntutysendmail@gmail.com';   % sender email address
password = '123456789Aa';           % sender email password
setpref('Internet','SMTP_Server','smtp.gmail.com');

setpref('Internet','E_mail',mail);
setpref('Internet','SMTP_Username',mail);
setpref('Internet','SMTP_Password',password);
props = java.lang.System.getProperties;
props.setProperty('mail.smtp.auth','true');
props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
props.setProperty('mail.smtp.socketFactory.port','465');

sendmail(target,subject,msg)
end
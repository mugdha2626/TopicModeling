import React from 'react';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faCopyright } from '@fortawesome/free-solid-svg-icons';
import { faGithub } from '@fortawesome/free-brands-svg-icons';

const Footer = () => {
  return (
    <footer style={styles.footer}>
      <p style={styles.text}>
        <FontAwesomeIcon icon={faCopyright} /> <br/>All rights reserved.
      </p>
      <p style={styles.text}>
        Developed in collaboration with Dr. Javier Gomez-Lavin and the Purdue VRAI Lab.
      </p>
      <p style={styles.text}>
        <a
          href="https://www.vrai-lab.com"
          target="_blank"
          rel="noopener noreferrer"
          style={styles.link}
        >Visit the VRAI Lab
        </a>
      </p>
    <div style={styles.socialLinks}>
        <a
          href=""
          target="_blank"
          rel="noopener noreferrer"
          style={styles.socialLink}
        >
          <FontAwesomeIcon icon={faGithub} />
        </a>
      </div>
    </footer>
  );
};

const styles = {
  footer: {
    backgroundColor: '#121212',
    color: '#E0E0E0',
    padding: '20px',
    textAlign: 'center',
    borderTop: '1px solid #121212',
    marginTop: 'auto', // Push footer to bottom
  },
  text: {
    margin: '5px 0',
    fontSize: '0.7rem',
  },
  link: {
    color: '#6200EE',
    textDecoration: 'none',
    fontWeight: '500',
  },
  socialLinks: {
    marginTop: '10px',
  },
  socialLink: {
    color: '#E0E0E0',
    margin: '0 10px',
    fontSize: '1rem',
    transition: 'color 0.3s',
  },
  socialLinkHover: {
    color: '#4CAF50',
    textDecoration: 'none',
    fontWeight: '500',
    transition: 'color 0.3s',
    '&:hover': {
      color: '#FFFFFF',
    },
    }
};

export default Footer;